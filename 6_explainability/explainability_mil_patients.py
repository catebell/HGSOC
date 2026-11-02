import os
import sys
import joblib
from torch_geometric.loader import DataLoader

import config_file_tree

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.GAT import GAT
from models.PatientMIL_Enhanced_ModalityDropout import PatientMILModel
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from PatientBagDataset import PatientBagDataset
from train_eval_utils import pre_train_imputation_encoding
from utils_explainability import *

# Configuration & Toggle
GENERATE_PLOTS = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

GAT_MODELS_DIR = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
#GAT_MODELS_DIR = os.path.join(config_file_tree.TRAINED_MODELS_DIR, "chemiorefract_partial", "GAT")
MIL_MODELS_DIR = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.PATIENT_ENCODER_FOLDER)
MIL_MODELS_DIR = os.path.join(config_file_tree.TRAINED_MODELS_DIR, "02_11_2026_GAT_MIL_isFluidPenalty", "MIL")
rel_model_path = os.path.relpath(MIL_MODELS_DIR, config_file_tree.TRAINED_MODELS_DIR)

OUTPUT_DIR = os.path.join(config_file_tree.EXPLAINABILITY_DIR, "models_explained", rel_model_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIL_MODEL_PATH = os.path.join(MIL_MODELS_DIR, f"best_MIL_{config_file_tree.PATIENT_TARGET_COL}_{'withClin' if config_file_tree.USE_CLINICAL_DATA else 'noClin'}.pt")

GRAPHS_DIR = config_file_tree.PROCESSED_GRAPHS_DIR
METADATA_PATH = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
CLINICAL_PATH = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")
TEST_CSV_PATH = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")

SAMPLE_TARGET_COL = config_file_tree.SAMPLE_TARGET_COL
PATIENT_TARGET_COL = config_file_tree.PATIENT_TARGET_COL
USE_CLINICAL = config_file_tree.USE_CLINICAL_DATA
GENE_MAPPING_FILE = os.path.join(config_file_tree.PROCESSED_SAMPLES_DIR, "gene_node_mapping.tsv")


def main():
    logging.info("Starting Patient MIL Explainability Analysis...")
    logging.info(f"Model Checkpoint Path: {to_rel_path(MIL_MODEL_PATH)}")
    logging.info(f"Explainability Output Directory: {to_rel_path(OUTPUT_DIR)}")

    df_metadata = pd.read_csv(METADATA_PATH, sep='\t')
    df_test_samples = pd.read_csv(TEST_CSV_PATH)
    df_metadata_test = df_metadata[df_metadata['sample_id'].isin(df_test_samples['sample_id'])].dropna(subset=[SAMPLE_TARGET_COL]).copy()

    df_clinical = pd.read_csv(CLINICAL_PATH, sep='\t')
    clinical_patient_id_col = 'patient_id' if 'patient_id' in df_clinical.columns else df_clinical.columns[0]
    df_clinical_clean = df_clinical.dropna(subset=[PATIENT_TARGET_COL]).copy()

    patient_label_encoder_path = os.path.join(MIL_MODELS_DIR, f"label_encoder_patient_{PATIENT_TARGET_COL}.joblib")
    if os.path.exists(patient_label_encoder_path):
        label_encoder_patient = joblib.load(patient_label_encoder_path)
        df_clinical_clean[PATIENT_TARGET_COL] = label_encoder_patient.transform(df_clinical_clean[PATIENT_TARGET_COL])
        patient_class_names = [str(c) for c in label_encoder_patient.classes_]
    else:
        patient_class_names = [str(c) for c in df_clinical_clean[PATIENT_TARGET_COL].unique()]

    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[PATIENT_TARGET_COL].to_dict()

    sample_label_encoder_path = os.path.join(GAT_MODELS_DIR, f"label_encoder_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(sample_label_encoder_path):
        label_encoder_sample = joblib.load(sample_label_encoder_path)
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_test[SAMPLE_TARGET_COL])
        num_sample_classes = len(label_encoder_sample.classes_)
    else:
        label_encoder_sample = joblib.LabelEncoder()
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.fit_transform(df_metadata_test[SAMPLE_TARGET_COL])
        num_sample_classes = len(label_encoder_sample.classes_)

    patient_clinical_map = None
    clinical_feature_cols = []
    num_clinical_features = 0

    if USE_CLINICAL:
        clin_prep_path = os.path.join(MIL_MODELS_DIR, f"clinical_preprocessors_{PATIENT_TARGET_COL}.joblib")
        if os.path.exists(clin_prep_path):
            p_dict = joblib.load(clin_prep_path)
            df_clinical_proc = pre_train_imputation_encoding(
                df=df_clinical_clean, task='classification', target_col=PATIENT_TARGET_COL, is_train=False,
                encoders=p_dict['encoders'], scalers=p_dict['scalers'], imputers=p_dict['imputers'],
                categories=p_dict['categories']
            ).set_index(clinical_patient_id_col)
            clinical_feature_cols = [c for c in df_clinical_proc.columns if c not in {PATIENT_TARGET_COL, 'patient_id'}]
            num_clinical_features = len(clinical_feature_cols)
            patient_clinical_map = {
                pid: torch.tensor(df_clinical_proc.loc[pid, clinical_feature_cols].values.astype(np.float32), dtype=torch.float32)
                for pid in df_clinical_proc.index
            }

    raw_sample_dataset = PatientSampleDataset(graphs_dir=GRAPHS_DIR, samples_metadata_df=df_metadata_test, task=config_file_tree.TASK, target_col=SAMPLE_TARGET_COL)
    sample_dataset = [raw_sample_dataset[i] for i in range(len(raw_sample_dataset))]

    node_scaler_path = os.path.join(GAT_MODELS_DIR, f"node_scaler_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(node_scaler_path):
        node_scaler = joblib.load(node_scaler_path)
        for graph in sample_dataset:
            graph.x = torch.tensor(node_scaler.transform(graph.x.numpy()), dtype=torch.float32)

    num_sample_features = 0
    if config_file_tree.USE_SAMPLES_METADATA:
        meta_prep_path = os.path.join(GAT_MODELS_DIR, f"metadata_preprocessors_{SAMPLE_TARGET_COL}.joblib")
        if os.path.exists(meta_prep_path):
            meta_prep = joblib.load(meta_prep_path)
            df_metadata_test_proc = pre_train_imputation_encoding(
                df=df_metadata_test, task=config_file_tree.TASK, target_col=SAMPLE_TARGET_COL, is_train=False,
                encoders=meta_prep['encoders'], scalers=meta_prep['scalers'], imputers=meta_prep['imputers'],
                categories=meta_prep['categories']
            ).set_index('sample_id')
            sample_feature_cols = [c for c in df_metadata_test_proc.columns if c not in {SAMPLE_TARGET_COL, 'patient_id'}]
            num_sample_features = len(sample_feature_cols)
            update_graphs_metadata(sample_dataset, df_metadata_test_proc, sample_feature_cols)

    patient_test_dataset = PatientBagDataset(sample_dataset, patient_target_map, patient_clinical_map=patient_clinical_map if USE_CLINICAL else None)
    patients_test_loader = DataLoader(patient_test_dataset, batch_size=4, shuffle=False)

    sample = raw_sample_dataset[0]
    num_genes_features = sample.x.size(1)
    num_unique_genes = sample.x.size(0)

    model_samples_encoder = GAT(
        num_node_features=num_genes_features, num_classes=num_sample_classes, num_nodes=num_unique_genes,
        num_sample_features=num_sample_features if config_file_tree.USE_SAMPLES_METADATA else 0
    )
    model_patient = PatientMILModel(
        sample_encoder=model_samples_encoder, num_classes=len(patient_class_names),
        num_tissue_classes=num_sample_classes, num_clinical_features=num_clinical_features
    ).to(device)

    if os.path.exists(MIL_MODEL_PATH):
        model_patient.load_state_dict(torch.load(MIL_MODEL_PATH, map_location=device))

    gene_map_dict = load_gene_mappings(GENE_MAPPING_FILE)
    feature_names = [config_file_tree.NODE_FEATURE_NAMES[i] for i in config_file_tree.NODE_FEATURE_INDEX]

    # 1. Generate TSV tabular outputs
    extract_mil_edge_attention(model_patient, patients_test_loader, gene_map_dict, num_unique_genes, OUTPUT_DIR, device)
    extract_gene_saliency_and_attention_map(model_patient, patients_test_loader, gene_map_dict, num_unique_genes, feature_names, OUTPUT_DIR, device, prefix="mil")
    compute_feature_importance(model_patient, patients_test_loader, clinical_feature_cols, OUTPUT_DIR, device, is_mil=True, prefix="mil", feature_type_label="clinical")
    compute_shap_values(model_patient, patients_test_loader, feature_names, clinical_feature_cols, OUTPUT_DIR, device, gene_map_dict, num_unique_genes, is_mil=True, prefix="mil", generate_plots=GENERATE_PLOTS, top_n=20)
    extract_mil_tissue_attention_aggregation(model_patient, patients_test_loader, OUTPUT_DIR, device,
                                             label_encoder_sample=label_encoder_sample)

    # 2. Optionally generate visual plots
    if GENERATE_PLOTS:
        logging.info("Generating MIL visual plots...")
        edge_tsv_path = os.path.join(OUTPUT_DIR, get_edge_attention_tsv_filename("mil"))
        plot_interactive_gene_network(tsv_path=edge_tsv_path, output_html=os.path.join(OUTPUT_DIR, get_net_interactive_html_filename("mil")), top_n=100)
        plot_static_gene_network(tsv_path=edge_tsv_path, output_png=os.path.join(OUTPUT_DIR, get_net_static_png_filename("mil")), top_n=100)

        clin_imp_path = os.path.join(OUTPUT_DIR, get_importance_tsv_filename("mil"))
        if os.path.exists(clin_imp_path):
            df_clin_imp = pd.read_csv(clin_imp_path, sep='\t')
            plot_importance_barplot(df_clin_imp, x_col='importance', y_col='feature', title="Patient Clinical Features Importance", output_path=os.path.join(OUTPUT_DIR, get_importance_png_filename("mil")), orientation='vertical')

        sal_path = os.path.join(OUTPUT_DIR, get_gene_attention_map_tsv_filename("mil"))
        if os.path.exists(sal_path):
            df_mil_sal = pd.read_csv(sal_path, sep='\t')
            top_5_saliency = df_mil_sal.head(5)
            top_5_genes_dict = dict(zip(top_5_saliency['gene_id'], top_5_saliency['gene_name']))
            ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}

            records = collect_top_genes_node_features(patients_test_loader, top_5_genes_dict.keys(), ensg_to_idx, num_unique_genes, is_mil=True)
            plot_gene_feature_boxplots(records, top_5_genes_dict, class_names=patient_class_names, output_dir=OUTPUT_DIR, prefix="mil", feature_names=feature_names)
            plot_gene_feature_attention_heatmap(tsv_path=sal_path, output_png=os.path.join(OUTPUT_DIR, get_heatmap_png_filename("mil")), feature_names=feature_names, top_n=25)

    logging.info("Patient MIL Explainability Analysis Complete!\n")

    if os.path.exists(sal_path):
        run_posthoc_pathway_enrichment(saliency_tsv_path=sal_path, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()