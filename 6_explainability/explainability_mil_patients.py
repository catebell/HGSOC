import os
import sys
import argparse
import joblib
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

import config_file_tree

training_dir = os.path.join(config_file_tree.ROOT_DIR, "5_training")
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

try:
    import train_functions
    sys.modules["train_functions"] = train_functions
except ImportError:
    pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.GAT import GAT
from models.PatientMIL_regularized import PatientMILModel
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from PatientBagDataset import PatientBagDataset
from utils_train_eval import pre_train_imputation_encoding
from utils_explainability import *

GENERATE_PLOTS = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint_arg(arg_value, default_folder_name):
    """Resolve a CLI argument (.pt file, directory, or folder name inside TRAINED_MODELS_DIR)."""
    if arg_value:
        if os.path.isfile(arg_value):
            abs_pt = os.path.abspath(arg_value)
            return os.path.dirname(abs_pt), abs_pt
        if os.path.isdir(arg_value):
            return os.path.abspath(arg_value), None
        candidate_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, arg_value)
        if os.path.isdir(candidate_dir):
            return os.path.abspath(candidate_dir), None

    fallback_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, default_folder_name)
    return os.path.abspath(fallback_dir), None


def main():
    parser = argparse.ArgumentParser(description="Run MIL Patient Explainability Analysis")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "metrics", "importance", "shap", "graph", "boxplots", "enrichment", "tissue_attn", "pca"],
        help="Specific explainability step to execute",
    )
    parser.add_argument("--gat_model_path", type=str, default=None, help="Path to the paired GAT .pt checkpoint or directory")
    parser.add_argument("--mil_model_path", type=str, default=None, help="Path to the selected MIL .pt checkpoint or directory")

    args = parser.parse_args()
    steps_to_run = resolve_step_sequence(args.step, model_type="mil")
    logging.info(f"Resolved MIL execution sequence for '--step {args.step}': {steps_to_run}")

    MIL_MODELS_DIR, MIL_MODEL_PATH = _resolve_checkpoint_arg(
        args.mil_model_path, config_file_tree.PATIENT_ENCODER_FOLDER
    )

    mil_cfg = load_model_training_config(MIL_MODELS_DIR)
    TASK = mil_cfg["TASK"]
    SAMPLE_TARGET_COL = mil_cfg["SAMPLE_TARGET_COL"]
    PATIENT_TARGET_COL = mil_cfg["PATIENT_TARGET_COL"]
    USE_CLINICAL = mil_cfg["USE_CLINICAL_DATA"]
    USE_SAMPLES_METADATA = mil_cfg["USE_SAMPLES_METADATA"]
    NODE_FEATURE_INDEX = mil_cfg["NODE_FEATURE_INDEX"]
    NODE_FEATURE_NAMES = mil_cfg["NODE_FEATURE_NAMES"]

    if MIL_MODEL_PATH is None:
        MIL_MODEL_PATH = os.path.join(
            MIL_MODELS_DIR,
            f"best_MIL_{PATIENT_TARGET_COL}_{'withClin' if USE_CLINICAL else 'noClin'}.pt",
        )

    sibling_gat_dir = os.path.join(os.path.dirname(MIL_MODELS_DIR), "GAT")
    if os.path.isdir(sibling_gat_dir) and any(f.endswith((".pt", ".joblib")) for f in os.listdir(sibling_gat_dir)):
        GAT_MODELS_DIR = sibling_gat_dir
    else:
        GAT_MODELS_DIR, _ = _resolve_checkpoint_arg(args.gat_model_path, config_file_tree.SAMPLE_ENCODER_FOLDER)

    OUTPUT_DIR = os.path.join(MIL_MODELS_DIR, "explained")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    GRAPHS_DIR, SPLIT_DIR, PROCESSED_SAMPLES_DIR = resolve_experiment_data_dirs(mil_cfg)
    METADATA_PATH = os.path.join(SPLIT_DIR, "metadata_for_training.tsv")
    CLINICAL_PATH = os.path.join(SPLIT_DIR, "clinical_data_for_training.tsv")
    TEST_CSV_PATH = os.path.join(SPLIT_DIR, "samples_test.csv")
    GENE_MAPPING_FILE = os.path.join(PROCESSED_SAMPLES_DIR, "gene_node_mapping.tsv")

    STEP_WEIGHTS = {
        "init": 12,
        "metrics": 5,
        "graph": 20,
        "importance": 12,
        "shap": 25,
        "enrichment": 20,
        "tissue_attn": 6,
        "pca": 25,
    }
    ProgressTracker.configure({k: STEP_WEIGHTS[k] for k in ["init"] + steps_to_run})
    ProgressTracker.start_phase("init", "Loading graphs & model checkpoints...")

    logging.info(f"Starting Patient MIL Explainability Analysis (Steps: {[s.upper() for s in steps_to_run]})...")
    logging.info(f"Model Checkpoint Path: {to_rel_path(MIL_MODEL_PATH)}")
    logging.info(f"Explainability Output Directory: {to_rel_path(OUTPUT_DIR)}")

    df_metadata = pd.read_csv(METADATA_PATH, sep="\t")
    df_test_samples = pd.read_csv(TEST_CSV_PATH)
    df_metadata_test = df_metadata[df_metadata["sample_id"].isin(df_test_samples["sample_id"])].dropna(subset=[SAMPLE_TARGET_COL]).copy()

    df_clinical = pd.read_csv(CLINICAL_PATH, sep="\t")
    clinical_patient_id_col = "patient_id" if "patient_id" in df_clinical.columns else df_clinical.columns[0]
    df_clinical_clean = df_clinical.dropna(subset=[PATIENT_TARGET_COL]).copy()

    sample_label_encoder_path = os.path.join(GAT_MODELS_DIR, f"label_encoder_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(sample_label_encoder_path):
        label_encoder_sample = joblib.load(sample_label_encoder_path)
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_test[SAMPLE_TARGET_COL])
    else:
        label_encoder_sample = LabelEncoder()
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.fit_transform(df_metadata_test[SAMPLE_TARGET_COL])
    sample_class_names = [str(c) for c in label_encoder_sample.classes_]
    num_sample_classes = len(sample_class_names)

    patient_label_encoder_path = os.path.join(MIL_MODELS_DIR, f"label_encoder_patient_{PATIENT_TARGET_COL}.joblib")
    if os.path.exists(patient_label_encoder_path):
        label_encoder_patient = joblib.load(patient_label_encoder_path)
        df_clinical_clean[PATIENT_TARGET_COL] = label_encoder_patient.transform(df_clinical_clean[PATIENT_TARGET_COL])
    else:
        label_encoder_patient = LabelEncoder()
        df_clinical_clean[PATIENT_TARGET_COL] = label_encoder_patient.fit_transform(df_clinical_clean[PATIENT_TARGET_COL])
    patient_class_names = [str(c) for c in label_encoder_patient.classes_]

    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[PATIENT_TARGET_COL].to_dict()

    patient_clinical_map = None
    clinical_feature_cols = []
    num_clinical_features = 0
    if USE_CLINICAL:
        clin_prep_path = os.path.join(MIL_MODELS_DIR, f"clinical_preprocessors_{PATIENT_TARGET_COL}.joblib")
        if os.path.exists(clin_prep_path):
            p_dict = joblib.load(clin_prep_path)
            df_clinical_proc = pre_train_imputation_encoding(
                df=df_clinical_clean, task="classification", target_col=PATIENT_TARGET_COL, is_train=False,
                encoders=p_dict["encoders"], scalers=p_dict["scalers"], imputers=p_dict["imputers"],
                categories=p_dict["categories"],
            ).set_index(clinical_patient_id_col)
            clinical_feature_cols = [c for c in df_clinical_proc.columns if c not in {PATIENT_TARGET_COL, "patient_id"}]
            num_clinical_features = len(clinical_feature_cols)
            patient_clinical_map = {
                pid: torch.tensor(df_clinical_proc.loc[pid, clinical_feature_cols].values.astype(np.float32), dtype=torch.float32)
                for pid in df_clinical_proc.index
            }

    raw_sample_dataset = PatientSampleDataset(graphs_dir=GRAPHS_DIR, samples_metadata_df=df_metadata_test, task=TASK, target_col=SAMPLE_TARGET_COL)
    sample_dataset = [raw_sample_dataset[i] for i in tracked_tqdm(range(len(raw_sample_dataset)), desc="Loading Test Sample Graphs")]

    node_scaler_path = os.path.join(GAT_MODELS_DIR, f"node_scaler_{SAMPLE_TARGET_COL}.joblib")
    if node_scaler_path and os.path.exists(node_scaler_path):
        node_scaler = joblib.load(node_scaler_path)
        for graph in sample_dataset:
            graph.x = torch.tensor(node_scaler.transform(graph.x.numpy()), dtype=torch.float32)

    num_sample_features = 0
    sample_feature_cols = []
    meta_prep_path = os.path.join(GAT_MODELS_DIR, f"metadata_preprocessors_{SAMPLE_TARGET_COL}.joblib")
    if USE_SAMPLES_METADATA and os.path.exists(meta_prep_path):
        meta_prep = joblib.load(meta_prep_path)
        df_metadata_test_proc = pre_train_imputation_encoding(
            df=df_metadata_test, task=TASK, target_col=SAMPLE_TARGET_COL, is_train=False,
            encoders=meta_prep["encoders"], scalers=meta_prep["scalers"], imputers=meta_prep["imputers"],
            categories=meta_prep["categories"],
        ).set_index("sample_id")
        sample_feature_cols = [c for c in df_metadata_test_proc.columns if c not in {SAMPLE_TARGET_COL, "patient_id"}]
        num_sample_features = len(sample_feature_cols)
        update_graphs_metadata(sample_dataset, df_metadata_test_proc, sample_feature_cols)

    patient_test_dataset = PatientBagDataset(
        sample_dataset, patient_target_map, patient_clinical_map=patient_clinical_map if USE_CLINICAL else None
    )
    patients_test_loader = DataLoader(patient_test_dataset, batch_size=4, shuffle=False)

    sample = raw_sample_dataset[0]
    num_genes_features = sample.x.size(1)
    num_unique_genes = sample.x.size(0)

    model_samples_encoder = GAT(
        num_node_features=num_genes_features,
        num_classes=num_sample_classes,
        num_nodes=num_unique_genes,
        num_sample_features=num_sample_features if USE_SAMPLES_METADATA else 0,
    )
    model_patient = PatientMILModel(
        sample_encoder=model_samples_encoder,
        num_classes=len(patient_class_names),
        num_tissue_classes=num_sample_classes,
        num_clinical_features=num_clinical_features,
    ).to(device)

    if os.path.exists(MIL_MODEL_PATH):
        model_patient.load_state_dict(torch.load(MIL_MODEL_PATH, map_location=device))

    gene_map_dict = load_gene_mappings(GENE_MAPPING_FILE)
    feature_names = [NODE_FEATURE_NAMES[i] for i in NODE_FEATURE_INDEX if i < len(NODE_FEATURE_NAMES)]

    if "metrics" in steps_to_run:
        ProgressTracker.start_phase("metrics", "Step: Sensitivity & Specificity Metrics...")
        model_patient.eval()
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for item in patients_test_loader:
                p_batch, target_y = (
                    (item[0].to(device), item[1].to(device))
                    if isinstance(item, (list, tuple))
                    else (item.to(device), item.patient_y.to(device))
                )
                logits, _ = model_patient(p_batch)
                y_true_list.append(target_y.cpu().numpy().ravel())
                y_prob_list.append(torch.softmax(logits, dim=1).cpu().numpy())

        evaluate_and_plot_sensitivity_specificity(
            y_true=np.concatenate(y_true_list),
            y_probs=np.vstack(y_prob_list),
            class_names=patient_class_names,
            output_dir=OUTPUT_DIR,
            prefix="mil",
            title_suffix="MIL Test Set",
        )

    # --- STEP 1: GENE SALIENCY, PPI NETWORKS, HEATMAP & TOP-5 BOXPLOTS ---
    if "graph" in steps_to_run:
        ProgressTracker.start_phase("graph", "Step: Gene Saliency, PPI Networks & Profiles...")
        extract_gene_saliency_and_attention_map(
            model_patient, patients_test_loader, gene_map_dict, num_unique_genes, feature_names, OUTPUT_DIR, device, prefix="mil"
        )
        sal_dir = os.path.join(OUTPUT_DIR, "gene_saliency_attention")
        sal_path = os.path.join(sal_dir, get_gene_attention_map_tsv_filename("mil"))

        extract_mil_edge_attention(model_patient, patients_test_loader, gene_map_dict, num_unique_genes, OUTPUT_DIR, device)
        if GENERATE_PLOTS:
            net_dir = os.path.join(OUTPUT_DIR, "gene_networks")
            edge_tsv_path = os.path.join(net_dir, get_edge_attention_tsv_filename("mil"))
            plot_interactive_gene_network(
                tsv_path=edge_tsv_path,
                output_html=os.path.join(net_dir, get_net_interactive_html_filename("mil")),
                top_n=100,
                saliency_tsv_path=sal_path,
            )
            plot_static_gene_network(
                tsv_path=edge_tsv_path,
                output_png=os.path.join(net_dir, get_net_static_png_filename("mil")),
                top_n=100,
            )

            if os.path.exists(sal_path):
                df_mil_sal = pd.read_csv(sal_path, sep="\t")
                top_5_saliency = df_mil_sal.head(5)
                top_5_genes_dict = dict(zip(top_5_saliency["gene_id"], top_5_saliency["gene_name"]))
                ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}

                records = collect_top_genes_node_features(
                    patients_test_loader, top_5_genes_dict.keys(), ensg_to_idx, num_unique_genes, is_mil=True
                )
                plot_gene_feature_boxplots(
                    records, top_5_genes_dict, class_names=patient_class_names, output_dir=sal_dir,
                    prefix="mil", feature_names=feature_names,
                )
                plot_gene_feature_attention_heatmap(
                    tsv_path=sal_path,
                    output_png=os.path.join(sal_dir, get_heatmap_png_filename("mil")),
                    feature_names=feature_names,
                    top_n=25,
                )

    # --- STEP 2: FEATURE IMPORTANCE ---
    if "importance" in steps_to_run:
        ProgressTracker.start_phase("importance", "Step: Clinical Feature Importance...")
        compute_feature_importance(
            model_patient, patients_test_loader, clinical_feature_cols,
            OUTPUT_DIR, device, is_mil=True, prefix="mil",
            feature_type_label="clinical", n_repeats=10,
        )
        if GENERATE_PLOTS:
            imp_dir = os.path.join(OUTPUT_DIR, "feature_importance")
            meta_imp_path = os.path.join(imp_dir, get_importance_tsv_filename("mil"))
            if os.path.exists(meta_imp_path):
                df_meta_imp = pd.read_csv(meta_imp_path, sep="\t")
                plot_importance_barplot(
                    df_meta_imp, x_col="importance", y_col="feature",
                    title="Patient Clinical Features Importance",
                    output_path=os.path.join(imp_dir, get_importance_png_filename("mil")),
                    orientation="vertical",
                )

    # --- STEP 3: SHAP VALUES ---
    if "shap" in steps_to_run:
        ProgressTracker.start_phase("shap", "Step: SHAP Explanations...")
        compute_shap_values(
            model_patient, patients_test_loader, feature_names, clinical_feature_cols, OUTPUT_DIR, device,
            gene_map_dict, num_unique_genes, is_mil=True, prefix="mil", generate_plots=GENERATE_PLOTS, top_n=20,
        )

    # --- STEP 4: PATHWAY ENRICHMENT ---
    if "enrichment" in steps_to_run:
        ProgressTracker.start_phase("enrichment", "Step: Pathway Enrichment (Hallmark/KEGG/GSEA)...")
        sal_path = os.path.join(OUTPUT_DIR, "gene_saliency_attention", get_gene_attention_map_tsv_filename("mil"))
        shap_paths = [
            os.path.join(OUTPUT_DIR, "shap_analysis", get_gene_shap_tsv_filename("mil", fn))
            for fn in feature_names
        ]
        run_posthoc_pathway_enrichment(
            saliency_tsv_path=sal_path,
            output_dir=OUTPUT_DIR,
            shap_tsv_paths=shap_paths,
            directional_ora=(len(patient_class_names) == 2),
        )

    # --- STEP 5: MIL TISSUE ATTENTION AGGREGATION ---
    if "tissue_attn" in steps_to_run:
        ProgressTracker.start_phase("tissue_attn", "Step: Tissue Attention Aggregation...")
        extract_mil_tissue_attention_aggregation(
            model_patient, patients_test_loader, OUTPUT_DIR, device, label_encoder_sample=label_encoder_sample
        )

    # --- STEP 6: LATENT SPACE EMBEDDINGS PCA / UMAP ---
    if "pca" in steps_to_run:
        ProgressTracker.start_phase("pca", "Step: Embedding Latent Space (PCA/UMAP)...")
        df_metadata_full = df_metadata.dropna(subset=[SAMPLE_TARGET_COL]).copy()
        df_metadata_full[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_full[SAMPLE_TARGET_COL])

        cached_test_graphs = dict(zip(df_metadata_test["sample_id"].astype(str), sample_dataset))
        df_metadata_non_test = df_metadata_full[~df_metadata_full["sample_id"].astype(str).isin(cached_test_graphs)].copy()

        non_test_graphs = []
        if not df_metadata_non_test.empty:
            raw_non_test_ds = PatientSampleDataset(
                graphs_dir=GRAPHS_DIR, samples_metadata_df=df_metadata_non_test,
                task=TASK, target_col=SAMPLE_TARGET_COL,
            )
            non_test_graphs = [
                raw_non_test_ds[i]
                for i in tracked_tqdm(range(len(raw_non_test_ds)), desc="Loading Train/Val Graphs for PCA", start_frac=0.0, end_frac=0.45)
            ]

            if os.path.exists(node_scaler_path):
                scaled_cat = torch.as_tensor(
                    node_scaler.transform(torch.cat([g.x for g in non_test_graphs], dim=0).numpy()),
                    dtype=torch.float32,
                )
                for graph, x_sc in zip(non_test_graphs, torch.split(scaled_cat, [g.x.size(0) for g in non_test_graphs])):
                    graph.x = x_sc

            if USE_SAMPLES_METADATA and os.path.exists(meta_prep_path):
                df_non_test_proc = pre_train_imputation_encoding(
                    df=df_metadata_non_test, task=TASK, target_col=SAMPLE_TARGET_COL, is_train=False,
                    encoders=meta_prep["encoders"], scalers=meta_prep["scalers"], imputers=meta_prep["imputers"],
                    categories=meta_prep["categories"],
                ).set_index("sample_id")
                update_graphs_metadata(non_test_graphs, df_non_test_proc, sample_feature_cols)

        non_test_map = dict(zip(df_metadata_non_test["sample_id"].astype(str), non_test_graphs))
        sample_dataset_full = [
            cached_test_graphs[sid] if sid in cached_test_graphs else non_test_map[sid]
            for sid in df_metadata_full["sample_id"].astype(str)
        ]

        patient_full_dataset = PatientBagDataset(
            sample_dataset_full, patient_target_map,
            patient_clinical_map=patient_clinical_map if USE_CLINICAL else None,
        )
        patients_full_loader = DataLoader(patient_full_dataset, batch_size=16, shuffle=False)

        test_patient_ids = {str(g.patient_id).strip() for g in sample_dataset if hasattr(g, "patient_id")}
        ProgressTracker.update(0.60, "Computing MIL PCA & UMAP (All & Test)...", force=True)
        generate_mil_pca_plots(
            model_patient, patients_full_loader, OUTPUT_DIR, device,
            label_encoder_sample=label_encoder_sample,
            label_encoder_patient=label_encoder_patient,
            patient_target_map=patient_target_map,
            suffix="all",
            test_patient_ids=test_patient_ids,
        )
        ProgressTracker.finish("MIL PCA & UMAP completed")


if __name__ == "__main__":
    main()