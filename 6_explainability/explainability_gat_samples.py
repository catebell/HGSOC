import os
import sys
import joblib
from torch_geometric.loader import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from train_eval_utils import pre_train_imputation_encoding
from utils_explainability import *

# Configuration & Toggle
GENERATE_PLOTS = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Determine relative checkpoint directory inside trained_models and explainability output directory inside models_explained
TRAINED_MODELS_DIR = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
rel_model_path = os.path.relpath(TRAINED_MODELS_DIR, config_file_tree.TRAINED_MODELS_DIR)

OUTPUT_DIR = os.path.join(config_file_tree.EXPLAINABILITY_DIR, "models_explained", rel_model_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, f"best_GAT_{config_file_tree.SAMPLE_TARGET_COL}_{'withMeta' if config_file_tree.USE_SAMPLES_METADATA else 'noMeta'}.pt")
GRAPHS_DIR = config_file_tree.PROCESSED_GRAPHS_DIR
METADATA_PATH = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
TEST_CSV_PATH = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")

SAMPLE_TARGET_COL = config_file_tree.SAMPLE_TARGET_COL
USE_METADATA = config_file_tree.USE_SAMPLES_METADATA
GENE_MAPPING_FILE = os.path.join(config_file_tree.PROCESSED_SAMPLES_DIR, "gene_node_mapping.tsv")


def main():
    logging.info("Starting GAT Explainability Analysis...")
    logging.info(f"Model Checkpoint Path: {to_rel_path(MODEL_PATH)}")
    logging.info(f"Explainability Output Directory: {to_rel_path(OUTPUT_DIR)}")

    df_metadata = pd.read_csv(METADATA_PATH, sep='\t')
    df_test_samples = pd.read_csv(TEST_CSV_PATH)
    df_metadata_test = df_metadata[df_metadata['sample_id'].isin(df_test_samples['sample_id'])].dropna(subset=[SAMPLE_TARGET_COL]).copy()

    label_encoder_path = os.path.join(TRAINED_MODELS_DIR, f"label_encoder_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(label_encoder_path):
        label_encoder_sample = joblib.load(label_encoder_path)
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_test[SAMPLE_TARGET_COL])
        sample_class_names = [str(c) for c in label_encoder_sample.classes_]
    else:
        sample_class_names = df_metadata[SAMPLE_TARGET_COL].dropna().astype(str).unique().tolist()

    raw_dataset = PatientSampleDataset(graphs_dir=GRAPHS_DIR, samples_metadata_df=df_metadata_test, task=config_file_tree.TASK, target_col=SAMPLE_TARGET_COL)
    dataset = [raw_dataset[i] for i in range(len(raw_dataset))]

    node_scaler_path = os.path.join(TRAINED_MODELS_DIR, f"node_scaler_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(node_scaler_path):
        node_scaler = joblib.load(node_scaler_path)
        for graph in dataset:
            graph.x = torch.tensor(node_scaler.transform(graph.x.numpy()), dtype=torch.float32)

    metadata_cols = []
    num_sample_features = 0
    if USE_METADATA:
        meta_prep_path = os.path.join(TRAINED_MODELS_DIR, f"metadata_preprocessors_{SAMPLE_TARGET_COL}.joblib")
        if os.path.exists(meta_prep_path):
            prep_dict = joblib.load(meta_prep_path)
            df_test_proc = pre_train_imputation_encoding(
                df=df_metadata_test, task=config_file_tree.TASK, target_col=SAMPLE_TARGET_COL, is_train=False,
                encoders=prep_dict['encoders'], scalers=prep_dict['scalers'], imputers=prep_dict['imputers'],
                categories=prep_dict['categories']
            ).set_index('sample_id')
            metadata_cols = [c for c in df_test_proc.columns if c not in {SAMPLE_TARGET_COL, 'patient_id'}]
            num_sample_features = len(metadata_cols)
            update_graphs_metadata(dataset, df_test_proc, metadata_cols)

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    sample = dataset[0]
    num_genes_features = sample.x.size(1)
    num_unique_genes = sample.x.size(0)

    model = GAT(
        num_node_features=num_genes_features, num_classes=len(sample_class_names),
        num_nodes=num_unique_genes,
        num_sample_features=num_sample_features if config_file_tree.USE_SAMPLES_METADATA else 0
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    gene_map_dict = load_gene_mappings(GENE_MAPPING_FILE)
    feature_names = [config_file_tree.NODE_FEATURE_NAMES[i] for i in config_file_tree.NODE_FEATURE_INDEX]

    # 1. Generate TSV tabular outputs
    extract_gat_edge_attention(model, test_loader, gene_map_dict, num_unique_genes, OUTPUT_DIR, device)
    extract_gene_saliency_and_attention_map(model, test_loader, gene_map_dict, num_unique_genes, feature_names, OUTPUT_DIR, device, prefix="gat")
    compute_feature_importance(model, test_loader, metadata_cols, OUTPUT_DIR, device, is_mil=False, prefix="gat", feature_type_label="metadata")
    compute_shap_values(model, test_loader, feature_names, metadata_cols, OUTPUT_DIR, device, gene_map_dict, num_unique_genes, is_mil=False, prefix="gat", generate_plots=GENERATE_PLOTS, top_n=20)

    # 2. Optionally generate visual plots
    if GENERATE_PLOTS:
        logging.info("Generating GAT visual plots...")
        edge_tsv_path = os.path.join(OUTPUT_DIR, get_edge_attention_tsv_filename("gat"))
        plot_interactive_gene_network(tsv_path=edge_tsv_path, output_html=os.path.join(OUTPUT_DIR, get_net_interactive_html_filename("gat")), top_n=100)
        plot_static_gene_network(tsv_path=edge_tsv_path, output_png=os.path.join(OUTPUT_DIR, get_net_static_png_filename("gat")), top_n=100)

        meta_imp_path = os.path.join(OUTPUT_DIR, get_importance_tsv_filename("gat"))
        if os.path.exists(meta_imp_path):
            df_meta_imp = pd.read_csv(meta_imp_path, sep='\t')
            plot_importance_barplot(df_meta_imp, x_col='importance', y_col='feature', title="Sample Metadata Features Importance", output_path=os.path.join(OUTPUT_DIR, get_importance_png_filename("gat")), orientation='vertical')

        sal_path = os.path.join(OUTPUT_DIR, get_gene_attention_map_tsv_filename("gat"))
        if os.path.exists(sal_path):
            df_gat_sal = pd.read_csv(sal_path, sep='\t')
            top_5_saliency = df_gat_sal.head(5)
            top_5_genes_dict = dict(zip(top_5_saliency['gene_id'], top_5_saliency['gene_name']))
            ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}

            records = collect_top_genes_node_features(test_loader, top_5_genes_dict.keys(), ensg_to_idx, num_unique_genes, is_mil=False)
            plot_gene_feature_boxplots(records, top_5_genes_dict, class_names=sample_class_names, output_dir=OUTPUT_DIR, prefix="gat", feature_names=feature_names)
            plot_gene_feature_attention_heatmap(tsv_path=sal_path, output_png=os.path.join(OUTPUT_DIR, get_heatmap_png_filename("gat")), feature_names=feature_names, top_n=25)

    logging.info("GAT Sample Explainability Analysis Complete!\n")

    if os.path.exists(sal_path):
        run_posthoc_pathway_enrichment(saliency_tsv_path=sal_path, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()