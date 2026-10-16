import os
import logging
import joblib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

import config_file_tree
from models.GAT import GAT
from models.PatientMIL import PatientMILModel
from PatientSampleDataset import PatientSampleDataset
from PatientBagDataset import PatientBagDataset
from train_eval_utils import pre_train_imputation_encoding
from plot_utils import plot_gene_feature_boxplots, plot_importance_barplot, collect_top_genes_node_features

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Directories and Settings
models_date = "15_10_2026"  # Nome della cartella/data dei modelli salvati in trained_models/

OUTPUT_DIR = "mil_patients_explained"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIL_MODELS_DIR = os.path.join("trained_models", models_date, "MIL")
GAT_MODELS_DIR = os.path.join("trained_models", models_date, "GAT")

MIL_MODEL_PATH = os.path.join(MIL_MODELS_DIR, f"best_MIL_{config_file_tree.PATIENT_TARGET_COL}_{'withClin' if config_file_tree.USE_CLINICAL_DATA else 'noClin'}.pt")

GRAPHS_DIR = config_file_tree.PROCESSED_GRAPHS_DIR
METADATA_PATH = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
CLINICAL_PATH = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")
TEST_CSV_PATH = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")

SAMPLE_TARGET_COL = config_file_tree.SAMPLE_TARGET_COL
PATIENT_TARGET_COL = config_file_tree.PATIENT_TARGET_COL
USE_CLINICAL = config_file_tree.USE_CLINICAL_DATA

GENE_MAPPING_FILE = os.path.join(config_file_tree.PROCESSED_SAMPLES_DIR, "gene_node_mapping.tsv")


def load_gene_mappings():
    """Load gene node mapping generated during graph creation."""
    gene_map_dict = {}
    if os.path.exists(GENE_MAPPING_FILE):
        df_map = pd.read_csv(GENE_MAPPING_FILE, sep='\t')
        gene_map_dict = {
            int(row['node_index']): (str(row['gene_id']), str(row['gene_symbol']))
            for _, row in df_map.iterrows()
        }
        logging.info(f"Loaded {len(gene_map_dict)} gene node mappings from {GENE_MAPPING_FILE}")
    else:
        logging.warning(f"Gene mapping file {GENE_MAPPING_FILE} not found!")
    return gene_map_dict


def get_gene_name(gene_idx, gene_map_dict):
    """Retrieve gene symbol and gene_id directly from node index."""
    if gene_idx in gene_map_dict:
        return gene_map_dict[gene_idx]
    return f"GENE_{gene_idx}", f"Unknown_{gene_idx}"


def extract_mil_edge_attention(model, loader, gene_map_dict, num_unique_genes):
    """Extract weighted PPI edge attention combining MIL sample attention and GAT edge attention."""
    logging.info("Extracting MIL patient-level edge attention weights...")
    model.eval()

    edge_attention_sum = {}
    total_bags = 0

    with torch.no_grad():
        for patient_batch, target_y, patient_id in tqdm(loader, desc="MIL Edge Attention"):
            patient_batch = patient_batch.to(device)

            # 1. Attenzione MIL sui campioni del paziente
            logits, attn_weights = model(patient_batch)
            attn_weights = attn_weights.squeeze(0)  # Shape: [N_samples]

            # 2. Attenzione GAT sugli archi del tessuto
            sample_encoder = model.sample_encoder
            _, (edge_index, alpha) = sample_encoder.conv1(
                patient_batch.x, patient_batch.edge_index, return_attention_weights=True
            )
            mean_alpha = alpha.mean(dim=1)  # [E]

            # Pondera l'attenzione dell'arco con l'attenzione MIL del campione corrispondente
            sample_batch_ids = patient_batch.batch[edge_index[1]]
            sample_weights = attn_weights[sample_batch_ids]
            combined_alpha = (mean_alpha * sample_weights).cpu().numpy()

            src_nodes = (edge_index[0] % num_unique_genes).cpu().numpy()
            dst_nodes = (edge_index[1] % num_unique_genes).cpu().numpy()

            for i in range(len(combined_alpha)):
                u, v = int(src_nodes[i]), int(dst_nodes[i])
                if u == v:
                    continue
                edge_key = tuple(sorted((u, v)))
                edge_attention_sum[edge_key] = edge_attention_sum.get(edge_key, 0.0) + float(combined_alpha[i])

            total_bags += 1

    results = []
    for (u, v), total_att in edge_attention_sum.items():
        avg_att = total_att / (total_bags + 1e-6)

        gene_id_a, symbol_a = get_gene_name(u, gene_map_dict)
        gene_id_b, symbol_b = get_gene_name(v, gene_map_dict)

        results.append({
            'gene_id_a': gene_id_a,
            'gene_symbol_a': symbol_a,
            'gene_id_b': gene_id_b,
            'gene_symbol_b': symbol_b,
            'interaction': f"{symbol_a} - {symbol_b}",
            'attention_weight': float(avg_att)
        })

    df_edges = pd.DataFrame(results).sort_values(by='attention_weight', ascending=False)
    out_path = os.path.join(OUTPUT_DIR, "mil_edge_attention.tsv")
    df_edges.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved MIL edge attention to {out_path}")
    return df_edges

def extract_mil_gene_saliency(model, loader, gene_map_dict, num_unique_genes):
    """Compute gene saliency scores driving patient prediction gradient."""
    logging.info("Extracting MIL gene saliency scores...")
    model.eval()

    saliency_sum = torch.zeros(num_unique_genes, device=device)
    gene_counts = torch.zeros(num_unique_genes, device=device)

    for patient_batch, target_y, patient_id in tqdm(loader, desc="MIL Saliency"):
        patient_batch = patient_batch.to(device)
        patient_batch.x.requires_grad = True

        logits, _ = model(patient_batch)
        pred_class = logits.argmax(dim=1)
        target_score = logits[0, pred_class]

        model.zero_grad()
        target_score.backward()

        node_saliency = (patient_batch.x.grad * patient_batch.x).abs().sum(dim=1)

        node_indices = torch.arange(patient_batch.x.size(0), device=device) % num_unique_genes
        saliency_sum.scatter_add_(0, node_indices, node_saliency)
        gene_counts.scatter_add_(0, node_indices, torch.ones_like(node_saliency))

    avg_saliency = (saliency_sum / (gene_counts + 1e-6)).detach().cpu().numpy()

    max_val = avg_saliency.max() if avg_saliency.max() > 0 else 1.0
    norm_saliency = avg_saliency / max_val

    results = []
    for idx in range(num_unique_genes):
        ensg_id, symbol = get_gene_name(idx, gene_map_dict)
        results.append({
            'gene_id': ensg_id,
            'gene_name': symbol,
            'saliency_score': float(norm_saliency[idx])
        })

    df_sal = pd.DataFrame(results).sort_values(by='saliency_score', ascending=False)
    out_path = os.path.join(OUTPUT_DIR, "mil_gene_saliency.tsv")
    df_sal.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved MIL gene saliency to {out_path}")


def compute_clinical_features_importance(model, loader, clinical_feature_cols):
    """Compute patient clinical feature importance using Permutation Importance."""
    logging.info("Computing patient clinical feature importance...")
    out_path = os.path.join(OUTPUT_DIR, "mil_clinical_features_importance.tsv")

    if not USE_CLINICAL or not clinical_feature_cols:
        df_empty = pd.DataFrame([{'feature': 'not used', 'importance': 'not used'}])
        df_empty.to_csv(out_path, sep='\t', index=False)
        logging.info(f"Clinical features not used. Wrote default file to {out_path}")
        return

    model.eval()

    def evaluate_acc(current_loader):
        correct, total = 0, 0
        with torch.no_grad():
            for patient_batch, target_y, _ in current_loader:
                patient_batch = patient_batch.to(device)
                target_y = target_y.to(device)
                logits, _ = model(patient_batch)
                preds = logits.argmax(dim=1)
                correct += int((preds == target_y).sum())
                total += 1
        return correct / total if total > 0 else 0.0

    baseline_acc = evaluate_acc(loader)
    feature_importances = []

    for col_idx, col_name in tqdm(enumerate(clinical_feature_cols), desc="Clinical Permutation"):
        permuted_correct, total = 0, 0
        with torch.no_grad():
            for patient_batch, target_y, _ in loader:
                patient_batch = patient_batch.to(device)
                target_y = target_y.to(device)

                if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
                    clinical_perm = patient_batch.clinical_x.clone()
                    perm_idx = torch.randperm(clinical_perm.size(0))
                    clinical_perm[:, col_idx] = clinical_perm[perm_idx, col_idx]
                    patient_batch.clinical_x = clinical_perm

                logits, _ = model(patient_batch)
                preds = logits.argmax(dim=1)
                permuted_correct += int((preds == target_y).sum())
                total += 1

        permuted_acc = permuted_correct / total if total > 0 else 0.0
        importance = baseline_acc - permuted_acc
        feature_importances.append({'feature': col_name, 'importance': float(importance)})

    df_imp = pd.DataFrame(feature_importances).sort_values(by='importance', ascending=False)
    df_imp.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved patient clinical feature importance to {out_path}")


def main():
    logging.info("Starting Patient MIL Explainability Analysis...")

    df_metadata = pd.read_csv(METADATA_PATH, sep='\t')
    df_test_samples = pd.read_csv(TEST_CSV_PATH)
    df_metadata_test = df_metadata[df_metadata['sample_id'].isin(df_test_samples['sample_id'])].dropna(subset=[SAMPLE_TARGET_COL]).copy()

    # 1. Load Clinical Data & Patient Label Encoder
    df_clinical = pd.read_csv(CLINICAL_PATH, sep='\t')
    clinical_patient_id_col = 'patient_id' if 'patient_id' in df_clinical.columns else df_clinical.columns[0]
    df_clinical_clean = df_clinical.dropna(subset=[PATIENT_TARGET_COL]).copy()

    patient_label_encoder_path = os.path.join(MIL_MODELS_DIR, f"label_encoder_patient_{PATIENT_TARGET_COL}.joblib")
    if os.path.exists(patient_label_encoder_path):
        label_encoder_patient = joblib.load(patient_label_encoder_path)
        df_clinical_clean[PATIENT_TARGET_COL] = label_encoder_patient.transform(df_clinical_clean[PATIENT_TARGET_COL])
        patient_classes = label_encoder_patient.classes_
        patient_class_names = [str(c) for c in patient_classes]
        logging.info(f"Loaded patient label encoder for {PATIENT_TARGET_COL}: {patient_class_names}")
    else:
        patient_classes = df_clinical_clean[PATIENT_TARGET_COL].unique()
        patient_class_names = [str(c) for c in patient_classes]
        logging.warning("Patient label encoder file not found. Using raw labels.")

    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[PATIENT_TARGET_COL].to_dict()

    # 2. Process Clinical Features (if enabled)
    patient_clinical_map = None
    clinical_feature_cols = []
    num_clinical_features = 0

    if USE_CLINICAL:
        clin_prep_path = os.path.join(MIL_MODELS_DIR, f"clinical_preprocessors_{PATIENT_TARGET_COL}.joblib")
        if os.path.exists(clin_prep_path):
            p_dict = joblib.load(clin_prep_path)
            df_clinical_proc = pre_train_imputation_encoding(
                df=df_clinical_clean,
                task='classification',
                target_col=PATIENT_TARGET_COL,
                is_train=False,
                encoders=p_dict['encoders'],
                scalers=p_dict['scalers'],
                imputers=p_dict['imputers'],
                categories=p_dict['categories']
            )
            df_clinical_proc = df_clinical_proc.set_index(clinical_patient_id_col)
            exclude_patients_cols = {PATIENT_TARGET_COL, 'patient_id'}
            clinical_feature_cols = [c for c in df_clinical_proc.columns if c not in exclude_patients_cols]
            num_clinical_features = len(clinical_feature_cols)

            patient_clinical_map = {}
            for patient_id in df_clinical_proc.index:
                feats = df_clinical_proc.loc[patient_id, clinical_feature_cols].values.astype(np.float32)
                patient_clinical_map[patient_id] = torch.tensor(feats, dtype=torch.float32)

            logging.info(f"Processed clinical features ({num_clinical_features}): {clinical_feature_cols}")
        else:
            logging.warning(f"Clinical preprocessor file {clin_prep_path} not found.")

    # 3. Load Sample Dataset & Apply Node Scaler
    sample_dataset = PatientSampleDataset(
        graphs_dir=GRAPHS_DIR,
        samples_metadata_df=df_metadata_test,
        task=config_file_tree.TASK,
        target_col=SAMPLE_TARGET_COL
    )

    node_scaler_path = os.path.join(GAT_MODELS_DIR, f"node_scaler_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(node_scaler_path):
        node_scaler = joblib.load(node_scaler_path)
        for graph in sample_dataset:
            scaled_x = node_scaler.transform(graph.x.numpy())
            graph.x = torch.tensor(scaled_x, dtype=torch.float32)
        logging.info("Applied node scaler to graph node features.")

    test_samples_list = [sample_dataset[i] for i in range(len(sample_dataset))]

    # 4. Patient Bag Dataset Initialization
    patient_test_dataset = PatientBagDataset(
        test_samples_list,
        patient_target_map,
        patient_clinical_map=patient_clinical_map if USE_CLINICAL else None
    )
    patients_test_loader = DataLoader(patient_test_dataset, batch_size=1, shuffle=False)

    num_genes_features = sample_dataset.num_features
    num_unique_genes = sample_dataset[0].x.size(0)

    # 5. Initialize GAT and Patient MIL Models
    sample_target_encoder_path = os.path.join(GAT_MODELS_DIR, f"label_encoder_{SAMPLE_TARGET_COL}.joblib")
    num_sample_classes = len(joblib.load(sample_target_encoder_path).classes_) if os.path.exists(sample_target_encoder_path) else len(df_metadata[SAMPLE_TARGET_COL].dropna().unique())

    model_samples_encoder = GAT(
        num_node_features=num_genes_features,
        num_classes=num_sample_classes,
        hidden_channels=64,
        heads=2
    )

    model_patient = PatientMILModel(
        sample_encoder=model_samples_encoder,
        num_classes=len(patient_class_names),
        num_clinical_features=num_clinical_features
    ).to(device)

    if os.path.exists(MIL_MODEL_PATH):
        model_patient.load_state_dict(torch.load(MIL_MODEL_PATH, map_location=device))
        logging.info(f"Loaded trained Patient MIL model from {MIL_MODEL_PATH}")
    else:
        logging.warning(f"Model file {MIL_MODEL_PATH} not found. Running with initial weights.")

    gene_map_dict = load_gene_mappings()

    # 6. Generate Explainability Outputs
    df_mil_edges = extract_mil_edge_attention(model_patient, patients_test_loader, gene_map_dict, num_unique_genes)

    extract_mil_gene_saliency(model_patient, patients_test_loader, gene_map_dict, num_unique_genes)
    compute_clinical_features_importance(model_patient, patients_test_loader, clinical_feature_cols)

    # 7. Generate Visual Plots
    logging.info("Generating MIL visual plots...")

    df_clin_imp = pd.read_csv(os.path.join(OUTPUT_DIR, "mil_clinical_features_importance.tsv"), sep='\t')
    plot_importance_barplot(
        df_clin_imp, x_col='importance', y_col='feature',
        title="Patient Clinical Features Importance",
        output_path=os.path.join(OUTPUT_DIR, "mil_clinical_features_importance.png"),
        orientation='vertical'
    )

    df_mil_sal = pd.read_csv(os.path.join(OUTPUT_DIR, "mil_gene_saliency.tsv"), sep='\t')
    plot_importance_barplot(
        df_mil_sal, x_col='saliency_score', y_col='gene_name',
        title="Top MIL Patient Gene Saliency Scores",
        output_path=os.path.join(OUTPUT_DIR, "mil_top_gene_saliency.png"),
        top_n=20
    )

    top_5_saliency = df_mil_sal.head(5)
    top_5_genes_dict = dict(zip(top_5_saliency['gene_id'], top_5_saliency['gene_name']))
    ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}

    records = collect_top_genes_node_features(
        patients_test_loader, top_5_genes_dict.keys(), ensg_to_idx, num_unique_genes, is_mil=True
    )

    plot_gene_feature_boxplots(
        records, top_5_genes_dict,
        class_names=patient_class_names,
        output_dir=OUTPUT_DIR,
        prefix="mil"
    )

    logging.info("Patient MIL Explainability Analysis and Plotting Complete!\n")


if __name__ == "__main__":
    main()