import os
import sys
import argparse
import joblib
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
import config_file_tree

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

training_dir = os.path.join(config_file_tree.ROOT_DIR, "5_training")
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

try:
    import train_functions
    sys.modules["train_functions"] = train_functions
except ImportError:
    pass

from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
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
    parser = argparse.ArgumentParser(description="Run GAT Explainability Analysis")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "metrics", "importance", "shap", "graph", "boxplots", "enrichment", "pca"],
        help="Specific explainability step to execute",
    )
    parser.add_argument(
        "--gat_model_path",
        type=str,
        default=None,
        help="Path to the selected GAT .pt checkpoint or directory",
    )

    args = parser.parse_args()
    steps_to_run = resolve_step_sequence(args.step, model_type="gat")
    logging.info(f"Resolved GAT execution sequence for '--step {args.step}': {steps_to_run}")

    TRAINED_MODELS_DIR, MODEL_PATH = _resolve_checkpoint_arg(
        args.gat_model_path, config_file_tree.SAMPLE_ENCODER_FOLDER
    )

    gat_cfg = load_model_training_config(TRAINED_MODELS_DIR)
    TASK = gat_cfg["TASK"]
    SAMPLE_TARGET_COL = gat_cfg["SAMPLE_TARGET_COL"]
    PATIENT_TARGET_COL = gat_cfg["PATIENT_TARGET_COL"]
    USE_METADATA = gat_cfg["USE_SAMPLES_METADATA"]
    NODE_FEATURE_INDEX = gat_cfg["NODE_FEATURE_INDEX"]
    NODE_FEATURE_NAMES = gat_cfg["NODE_FEATURE_NAMES"]

    if MODEL_PATH is None:
        MODEL_PATH = os.path.join(
            TRAINED_MODELS_DIR,
            f"best_GAT_{SAMPLE_TARGET_COL}_{'withMeta' if USE_METADATA else 'noMeta'}.pt",
        )

    OUTPUT_DIR = os.path.join(TRAINED_MODELS_DIR, "explained")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    GRAPHS_DIR, SPLIT_DIR, PROCESSED_SAMPLES_DIR = resolve_experiment_data_dirs(gat_cfg)
    METADATA_PATH = os.path.join(SPLIT_DIR, "metadata_for_training.tsv")
    TEST_CSV_PATH = os.path.join(SPLIT_DIR, "samples_test.csv")
    GENE_MAPPING_FILE = os.path.join(PROCESSED_SAMPLES_DIR, "gene_node_mapping.tsv")

    STEP_WEIGHTS = {
        "init": 12,
        "metrics": 5,
        "graph": 20,
        "importance": 12,
        "shap": 25,
        "enrichment": 20,
        "pca": 25,
    }
    ProgressTracker.configure({k: STEP_WEIGHTS[k] for k in ["init"] + steps_to_run})
    ProgressTracker.start_phase("init", "Loading graphs & model checkpoints...")

    logging.info(f"Starting GAT Explainability Analysis (Steps: {[s.upper() for s in steps_to_run]})...")
    logging.info(f"Model Checkpoint Path: {to_rel_path(MODEL_PATH)}")
    logging.info(f"Explainability Output Directory: {to_rel_path(OUTPUT_DIR)}")

    df_metadata = pd.read_csv(METADATA_PATH, sep="\t")
    df_test_samples = pd.read_csv(TEST_CSV_PATH)
    df_metadata_test = df_metadata[df_metadata["sample_id"].isin(df_test_samples["sample_id"])].dropna(subset=[SAMPLE_TARGET_COL]).copy()

    label_encoder_path = os.path.join(TRAINED_MODELS_DIR, f"label_encoder_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(label_encoder_path):
        label_encoder_sample = joblib.load(label_encoder_path)
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_test[SAMPLE_TARGET_COL])
    else:
        logging.warning(f"Sample label encoder not found at '{to_rel_path(label_encoder_path)}'. Using LabelEncoder() fallback.")
        label_encoder_sample = LabelEncoder()
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.fit_transform(df_metadata_test[SAMPLE_TARGET_COL])
    sample_class_names = [str(c) for c in label_encoder_sample.classes_]

    raw_dataset = PatientSampleDataset(graphs_dir=GRAPHS_DIR, samples_metadata_df=df_metadata_test, task=TASK, target_col=SAMPLE_TARGET_COL)
    dataset = [raw_dataset[i] for i in tracked_tqdm(range(len(raw_dataset)), desc="Loading Test Graphs")]

    node_scaler_path = os.path.join(TRAINED_MODELS_DIR, f"node_scaler_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(node_scaler_path):
        node_scaler = joblib.load(node_scaler_path)
        for graph in dataset:
            graph.x = torch.tensor(node_scaler.transform(graph.x.numpy()), dtype=torch.float32)

    metadata_cols = []
    num_sample_features = 0
    meta_prep_path = os.path.join(TRAINED_MODELS_DIR, f"metadata_preprocessors_{SAMPLE_TARGET_COL}.joblib")
    if USE_METADATA and os.path.exists(meta_prep_path):
        prep_dict = joblib.load(meta_prep_path)
        df_test_proc = pre_train_imputation_encoding(
            df=df_metadata_test, task=TASK, target_col=SAMPLE_TARGET_COL, is_train=False,
            encoders=prep_dict["encoders"], scalers=prep_dict["scalers"], imputers=prep_dict["imputers"],
            categories=prep_dict["categories"],
        ).set_index("sample_id")
        metadata_cols = [c for c in df_test_proc.columns if c not in {SAMPLE_TARGET_COL, "patient_id"}]
        num_sample_features = len(metadata_cols)
        update_graphs_metadata(dataset, df_test_proc, metadata_cols)

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    sample = dataset[0]
    num_genes_features = sample.x.size(1)
    num_unique_genes = sample.x.size(0)

    model = GAT(
        num_node_features=num_genes_features,
        num_classes=len(sample_class_names),
        num_nodes=num_unique_genes,
        num_sample_features=num_sample_features if USE_METADATA else 0,
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        logging.error(f"Sample Encoder Checkpoint not found at: {MODEL_PATH}")

    gene_map_dict = load_gene_mappings(GENE_MAPPING_FILE)
    feature_names = [NODE_FEATURE_NAMES[i] for i in NODE_FEATURE_INDEX if i < len(NODE_FEATURE_NAMES)]

    if "metrics" in steps_to_run:
        ProgressTracker.start_phase("metrics", "Step: Sensitivity & Specificity Metrics...")
        model.eval()
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(
                    batch.x, batch.edge_index, batch.batch,
                    metadata_x=getattr(batch, "metadata_x", None),
                )
                y_true_list.append(batch.y.cpu().numpy())
                y_prob_list.append(torch.softmax(logits, dim=1).cpu().numpy())

        evaluate_and_plot_sensitivity_specificity(
            y_true=np.concatenate(y_true_list),
            y_probs=np.vstack(y_prob_list),
            class_names=sample_class_names,
            output_dir=OUTPUT_DIR,
            prefix="gat",
            title_suffix="GAT Test Set",
        )

    # --- STEP 1: GENE SALIENCY, PPI NETWORKS, HEATMAP & TOP-5 BOXPLOTS ---
    if "graph" in steps_to_run:
        ProgressTracker.start_phase("graph", "Step: Gene Saliency, PPI Networks & Profiles...")
        extract_gene_saliency_and_attention_map(
            model, test_loader, gene_map_dict, num_unique_genes, feature_names, OUTPUT_DIR, device, prefix="gat"
        )
        sal_dir = os.path.join(OUTPUT_DIR, "gene_saliency_attention")
        sal_path = os.path.join(sal_dir, get_gene_attention_map_tsv_filename("gat"))

        extract_gat_edge_attention(model, test_loader, gene_map_dict, num_unique_genes, OUTPUT_DIR, device)
        if GENERATE_PLOTS:
            net_dir = os.path.join(OUTPUT_DIR, "gene_networks")
            edge_tsv_path = os.path.join(net_dir, get_edge_attention_tsv_filename("gat"))
            plot_interactive_gene_network(
                tsv_path=edge_tsv_path,
                output_html=os.path.join(net_dir, get_net_interactive_html_filename("gat")),
                top_n=100,
                saliency_tsv_path=sal_path,
            )
            plot_static_gene_network(
                tsv_path=edge_tsv_path,
                output_png=os.path.join(net_dir, get_net_static_png_filename("gat")),
                top_n=100,
            )

            if os.path.exists(sal_path):
                df_gat_sal = pd.read_csv(sal_path, sep="\t")
                top_5_saliency = df_gat_sal.head(5)
                top_5_genes_dict = dict(zip(top_5_saliency["gene_id"], top_5_saliency["gene_name"]))
                ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}

                records = collect_top_genes_node_features(
                    test_loader, top_5_genes_dict.keys(), ensg_to_idx, num_unique_genes, is_mil=False
                )
                plot_gene_feature_boxplots(
                    records, top_5_genes_dict, class_names=sample_class_names, output_dir=sal_dir,
                    prefix="gat", feature_names=feature_names,
                )
                plot_gene_feature_attention_heatmap(
                    tsv_path=sal_path,
                    output_png=os.path.join(sal_dir, get_heatmap_png_filename("gat")),
                    feature_names=feature_names,
                    top_n=25,
                )

    # --- STEP 2: FEATURE IMPORTANCE ---
    if "importance" in steps_to_run:
        ProgressTracker.start_phase("importance", "Step: Metadata Feature Importance...")
        compute_feature_importance(model, test_loader, metadata_cols, OUTPUT_DIR, device, is_mil=False, prefix="gat", feature_type_label="metadata")
        if GENERATE_PLOTS:
            imp_dir = os.path.join(OUTPUT_DIR, "feature_importance")
            meta_imp_path = os.path.join(imp_dir, get_importance_tsv_filename("gat"))
            if os.path.exists(meta_imp_path):
                df_meta_imp = pd.read_csv(meta_imp_path, sep="\t")
                plot_importance_barplot(
                    df_meta_imp, x_col="importance", y_col="feature",
                    title="Sample Metadata Features Importance",
                    output_path=os.path.join(imp_dir, get_importance_png_filename("gat")),
                    orientation="vertical",
                )

    # --- STEP 3: SHAP VALUES ---
    if "shap" in steps_to_run:
        ProgressTracker.start_phase("shap", "Step: SHAP Explanations...")
        compute_shap_values(
            model, test_loader, feature_names, metadata_cols, OUTPUT_DIR, device,
            gene_map_dict, num_unique_genes, is_mil=False, prefix="gat", generate_plots=GENERATE_PLOTS, top_n=20,
        )

    # --- STEP 4: PATHWAY ENRICHMENT ---
    if "enrichment" in steps_to_run:
        ProgressTracker.start_phase("enrichment", "Step: Pathway Enrichment (Hallmark/KEGG/GSEA)...")
        sal_path = os.path.join(OUTPUT_DIR, "gene_saliency_attention", get_gene_attention_map_tsv_filename("gat"))
        shap_paths = [
            os.path.join(OUTPUT_DIR, "shap_analysis", get_gene_shap_tsv_filename("gat", fn))
            for fn in feature_names
        ]
        run_posthoc_pathway_enrichment(
            saliency_tsv_path=sal_path,
            output_dir=OUTPUT_DIR,
            shap_tsv_paths=shap_paths,
            directional_ora=(len(sample_class_names) == 2),
        )

    # --- STEP 5: LATENT SPACE EMBEDDINGS PCA / UMAP ---
    if "pca" in steps_to_run:
        ProgressTracker.start_phase("pca", "Step: Embedding Latent Space (PCA/UMAP)...")
        df_metadata_full = df_metadata.dropna(subset=[SAMPLE_TARGET_COL]).copy()
        df_metadata_full[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_full[SAMPLE_TARGET_COL])

        cached_test_graphs = dict(zip(df_metadata_test["sample_id"].astype(str), dataset))
        df_metadata_non_test = df_metadata_full[~df_metadata_full["sample_id"].astype(str).isin(cached_test_graphs)].copy()

        non_test_graphs = []
        if not df_metadata_non_test.empty:
            raw_non_test_ds = PatientSampleDataset(
                graphs_dir=GRAPHS_DIR, samples_metadata_df=df_metadata_non_test,
                task=TASK, target_col=SAMPLE_TARGET_COL,
            )
            non_test_graphs = [
                raw_non_test_ds[i]
                for i in tracked_tqdm(range(len(raw_non_test_ds)), desc="Loading Train/Val Graphs for PCA", start_frac=0.0, end_frac=0.50)
            ]

            if os.path.exists(node_scaler_path):
                scaled_cat = torch.as_tensor(
                    node_scaler.transform(torch.cat([g.x for g in non_test_graphs], dim=0).numpy()),
                    dtype=torch.float32,
                )
                for graph, x_sc in zip(non_test_graphs, torch.split(scaled_cat, [g.x.size(0) for g in non_test_graphs])):
                    graph.x = x_sc

            if USE_METADATA and os.path.exists(meta_prep_path):
                df_non_test_proc = pre_train_imputation_encoding(
                    df=df_metadata_non_test, task=TASK, target_col=SAMPLE_TARGET_COL, is_train=False,
                    encoders=prep_dict["encoders"], scalers=prep_dict["scalers"], imputers=prep_dict["imputers"],
                    categories=prep_dict["categories"],
                ).set_index("sample_id")
                update_graphs_metadata(non_test_graphs, df_non_test_proc, metadata_cols)

        non_test_map = dict(zip(df_metadata_non_test["sample_id"].astype(str), non_test_graphs))
        dataset_full = [
            cached_test_graphs[sid] if sid in cached_test_graphs else non_test_map[sid]
            for sid in df_metadata_full["sample_id"].astype(str)
        ]
        full_loader = DataLoader(dataset_full, batch_size=16, shuffle=False)

        CLINICAL_PATH = os.path.join(SPLIT_DIR, "clinical_data_for_training.tsv")
        patient_target_map = None
        label_encoder_patient = None
        if os.path.exists(CLINICAL_PATH):
            df_clinical = pd.read_csv(CLINICAL_PATH, sep="\t")
            if PATIENT_TARGET_COL in df_clinical.columns:
                patient_target_map = df_clinical.set_index("patient_id")[PATIENT_TARGET_COL].to_dict()

        run_root = os.path.dirname(TRAINED_MODELS_DIR)
        for cand_dir in (
            os.path.join(run_root, "MIL"),
            os.path.join(run_root, "LinearMIL"),
            os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.PATIENT_ENCODER_FOLDER),
        ):
            le_path = os.path.join(cand_dir, f"label_encoder_patient_{PATIENT_TARGET_COL}.joblib")
            if os.path.exists(le_path):
                label_encoder_patient = joblib.load(le_path)
                break

        ProgressTracker.update(0.60, "Computing PCA & UMAP (All & Test)...", force=True)
        generate_gat_pca_plot(
            model, full_loader, OUTPUT_DIR, device,
            label_encoder_sample=label_encoder_sample,
            label_encoder_patient=label_encoder_patient,
            patient_target_map=patient_target_map,
            suffix="all",
            test_sample_ids=set(cached_test_graphs.keys()),
        )
        ProgressTracker.finish("PCA & UMAP completed")


if __name__ == "__main__":
    main()