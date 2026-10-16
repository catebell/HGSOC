import os
import logging
import joblib
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import config_file_tree
from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from train_eval_utils import pre_train_imputation_encoding
from plot_utils import plot_gene_feature_boxplots, plot_importance_barplot, collect_top_genes_node_features, \
    plot_interactive_gene_network, plot_static_gene_network

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Directories and Settings
models_date = "15_10_2026"  # Nome della cartella/data dei modelli salvati in trained_models/

OUTPUT_DIR = "gat_samples_explained"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINED_MODELS_DIR = os.path.join(config_file_tree.TRAINED_MODELS_DIR, models_date, "GAT")
MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, f"best_GAT_{config_file_tree.SAMPLE_TARGET_COL}_{'withMeta' if config_file_tree.USE_SAMPLES_METADATA else 'noMeta'}.pt")

GRAPHS_DIR = config_file_tree.PROCESSED_GRAPHS_DIR
METADATA_PATH = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
TEST_CSV_PATH = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")
SAMPLE_TARGET_COL = config_file_tree.SAMPLE_TARGET_COL
USE_METADATA = config_file_tree.USE_SAMPLES_METADATA

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


def extract_gat_edge_attention(model, loader, gene_map_dict, num_unique_genes):
    """Extract top gene-gene PPI interaction attention weights from GATv2Conv layer."""
    logging.info("Extracting GAT PPI edge attention weights...")
    model.eval()

    edge_attention_sum = {}
    edge_counts = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="GAT Edge Attention"):
            batch = batch.to(device)
            # Recupera pesi d'attenzione dalla convoluzione
            out, (edge_index, alpha) = model.conv1(batch.x, batch.edge_index, return_attention_weights=True)
            mean_alpha = alpha.mean(dim=1).cpu().numpy()  # Media tra le teste d'attenzione (heads)

            # Mappatura agli indici locali dei nodi [0 ... num_unique_genes - 1]
            src_nodes = (edge_index[0] % num_unique_genes).cpu().numpy()
            dst_nodes = (edge_index[1] % num_unique_genes).cpu().numpy()

            for i in range(len(mean_alpha)):
                u, v = int(src_nodes[i]), int(dst_nodes[i])

                # 1. Filtra i self-loop aggiunti dal GAT (nodo collegato a se stesso)
                if u == v:
                    continue

                # 2. Chiave unica per la coppia non orientata (A-B coincide con B-A)
                edge_key = tuple(sorted((u, v)))

                edge_attention_sum[edge_key] = edge_attention_sum.get(edge_key, 0.0) + float(mean_alpha[i])
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

    results = []
    for (u, v), total_att in edge_attention_sum.items():
        avg_att = total_att / edge_counts[(u, v)]

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
    out_path = os.path.join(OUTPUT_DIR, "gat_edge_attention.tsv")
    df_edges.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved GAT edge attention to {out_path}")
    return df_edges

def extract_gat_gene_saliency(model, loader, gene_map_dict, num_unique_genes):
    """Compute gene saliency scores driving prediction gradient."""
    logging.info("Extracting GAT gene saliency scores...")
    model.eval()

    saliency_sum = torch.zeros(num_unique_genes, device=device)
    gene_counts = torch.zeros(num_unique_genes, device=device)

    for batch in tqdm(loader, desc="GAT Saliency"):
        batch = batch.to(device)
        batch.x.requires_grad = True

        metadata_x = getattr(batch, 'metadata_x', None)
        logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_x)

        preds = logits.argmax(dim=1)
        target_scores = logits[range(len(preds)), preds].sum()

        model.zero_grad()
        target_scores.backward()

        node_saliency = (batch.x.grad * batch.x).abs().sum(dim=1)

        node_indices = torch.arange(batch.x.size(0), device=device) % num_unique_genes
        saliency_sum.scatter_add_(0, node_indices, node_saliency)
        gene_counts.scatter_add_(0, node_indices, torch.ones_like(node_saliency))

    avg_saliency = (saliency_sum / (gene_counts + 1e-6)).detach().cpu().numpy()

    max_val = avg_saliency.max() if avg_saliency.max() > 0 else 1.0
    norm_saliency = avg_saliency / max_val

    results = []
    for idx in range(num_unique_genes):
        if gene_counts[idx] > 0:
            ensg_id, symbol = get_gene_name(idx, gene_map_dict)
            results.append({
                'gene_id': ensg_id,
                'gene_name': symbol,
                'saliency_score': float(norm_saliency[idx])
            })

    df_sal = pd.DataFrame(results).sort_values(by='saliency_score', ascending=False)
    out_path = os.path.join(OUTPUT_DIR, "gat_gene_saliency.tsv")
    df_sal.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved GAT gene saliency to {out_path}")


def compute_sample_metadata_importance(model, loader, metadata_cols):
    """Compute sample metadata feature importance using Permutation Importance."""
    logging.info("Computing sample metadata feature importance...")
    out_path = os.path.join(OUTPUT_DIR, "gat_sample_features_importance.tsv")

    if not USE_METADATA or not metadata_cols:
        df_empty = pd.DataFrame([{'feature': 'not used', 'importance': 'not used'}])
        df_empty.to_csv(out_path, sep='\t', index=False)
        logging.info(f"Sample metadata not used. Wrote default file to {out_path}")
        return

    model.eval()

    def evaluate_acc(current_loader):
        correct, total = 0, 0
        with torch.no_grad():
            for batch in current_loader:
                batch = batch.to(device)
                metadata_x = getattr(batch, 'metadata_x', None)
                logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_x)
                preds = logits.argmax(dim=1)
                correct += int((preds == batch.y).sum())
                total += batch.num_graphs
        return correct / total if total > 0 else 0.0

    baseline_acc = evaluate_acc(loader)
    feature_importances = []

    for col_idx, col_name in tqdm(enumerate(metadata_cols), desc="Metadata Permutation"):
        permuted_correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                metadata_x = getattr(batch, 'metadata_x', None)
                if metadata_x is not None:
                    metadata_perm = metadata_x.clone()
                    perm_idx = torch.randperm(metadata_perm.size(0))
                    metadata_perm[:, col_idx] = metadata_perm[perm_idx, col_idx]
                else:
                    metadata_perm = None

                logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_perm)
                preds = logits.argmax(dim=1)
                permuted_correct += int((preds == batch.y).sum())
                total += batch.num_graphs

        permuted_acc = permuted_correct / total if total > 0 else 0.0
        importance = baseline_acc - permuted_acc
        feature_importances.append({'feature': col_name, 'importance': float(importance)})

    df_imp = pd.DataFrame(feature_importances).sort_values(by='importance', ascending=False)
    df_imp.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved sample metadata feature importance to {out_path}")


def main():
    logging.info("Starting GAT Explainability Analysis...")

    df_metadata = pd.read_csv(METADATA_PATH, sep='\t')
    df_test_samples = pd.read_csv(TEST_CSV_PATH)
    df_metadata_test = df_metadata[df_metadata['sample_id'].isin(df_test_samples['sample_id'])].dropna(
        subset=[SAMPLE_TARGET_COL]).copy()

    # 1. Load Target Label Encoder
    label_encoder_path = os.path.join(TRAINED_MODELS_DIR, f"label_encoder_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(label_encoder_path):
        label_encoder_sample = joblib.load(label_encoder_path)
        df_metadata_test[SAMPLE_TARGET_COL] = label_encoder_sample.transform(df_metadata_test[SAMPLE_TARGET_COL])
        sample_class_names = [str(c) for c in label_encoder_sample.classes_]
        logging.info(f"Loaded label encoder for {SAMPLE_TARGET_COL}: {sample_class_names}")
    else:
        sample_class_names = df_metadata[SAMPLE_TARGET_COL].dropna().astype(str).unique().tolist()
        logging.warning("Label encoder file not found. Using raw labels.")

    # 2. Sample Dataset Initialization & Pre-loading into RAM
    raw_dataset = PatientSampleDataset(
        graphs_dir=GRAPHS_DIR,
        samples_metadata_df=df_metadata_test,
        task=config_file_tree.TASK,
        target_col=SAMPLE_TARGET_COL
    )
    # Pre-caricamento in memoria RAM per mantenere persistenti le modifiche a metadata_x
    dataset = [raw_dataset[i] for i in range(len(raw_dataset))]

    # 3. Load and Apply Node Scaler
    node_scaler_path = os.path.join(TRAINED_MODELS_DIR, f"node_scaler_{SAMPLE_TARGET_COL}.joblib")
    if os.path.exists(node_scaler_path):
        node_scaler = joblib.load(node_scaler_path)
        for graph in dataset:
            scaled_x = node_scaler.transform(graph.x.numpy())
            graph.x = torch.tensor(scaled_x, dtype=torch.float32)
        logging.info("Applied node scaler to graph node features.")

    # 4. Process Sample Metadata Features (if enabled)
    metadata_cols = []
    num_sample_features = 0
    if USE_METADATA:
        meta_prep_path = os.path.join(TRAINED_MODELS_DIR,
                                      f"metadata_preprocessors_{SAMPLE_TARGET_COL}.joblib")
        if os.path.exists(meta_prep_path):
            prep_dict = joblib.load(meta_prep_path)
            df_test_proc = pre_train_imputation_encoding(
                df=df_metadata_test,
                task=config_file_tree.TASK,
                target_col=SAMPLE_TARGET_COL,
                is_train=False,
                encoders=prep_dict['encoders'],
                scalers=prep_dict['scalers'],
                imputers=prep_dict['imputers'],
                categories=prep_dict['categories']
            )
            df_test_proc = df_test_proc.set_index('sample_id')
            exclude_cols = {SAMPLE_TARGET_COL, 'patient_id'}
            metadata_cols = [c for c in df_test_proc.columns if c not in exclude_cols]
            num_sample_features = len(metadata_cols)

            # Ora dataset è una lista Python: le modifiche a graph.metadata_x rimangono in memoria!
            update_graphs_metadata(dataset, df_test_proc, metadata_cols)
            logging.info(f"Processed sample metadata features ({num_sample_features}): {metadata_cols}")
        else:
            logging.warning(f"Metadata preprocessor file {meta_prep_path} not found.")

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    num_genes_features = dataset[0].x.shape[1]
    sample = dataset[0]
    num_unique_genes = sample.x.size(0)

    # 5. Load Pre-trained GAT Model
    model = GAT(
        num_node_features=num_genes_features,
        num_classes=len(sample_class_names),
        hidden_channels=64,
        heads=2,
        num_sample_features=num_sample_features
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logging.info(f"Loaded trained GAT model from {MODEL_PATH}")
    else:
        logging.warning(f"Model file {MODEL_PATH} not found. Running with initial weights.")

    gene_map_dict = load_gene_mappings()

    # 6. Generate Explainability Outputs
    df_edges = extract_gat_edge_attention(model, test_loader, gene_map_dict, num_unique_genes)

    extract_gat_gene_saliency(model, test_loader, gene_map_dict, num_unique_genes)
    compute_sample_metadata_importance(model, test_loader, metadata_cols)

    # 7. Generate Visual Plots
    logging.info("Generating GAT visual plots...")

    edge_tsv_path = os.path.join(OUTPUT_DIR, "gat_edge_attention.tsv")

    # A. Grafo Interattivo HTML
    plot_interactive_gene_network(
        tsv_path=edge_tsv_path,
        output_html=os.path.join(
            OUTPUT_DIR, "gat_gene_network_interactive.html"
        ),
        top_n=50,
    )

    # B. Grafo Statico PNG
    plot_static_gene_network(
        tsv_path=os.path.join(OUTPUT_DIR, "gat_edge_attention.tsv"),
        output_png=os.path.join(OUTPUT_DIR, "gat_gene_network_static.png"),
        top_n=50
    )

    df_meta_imp = pd.read_csv(os.path.join(OUTPUT_DIR, "gat_sample_features_importance.tsv"), sep='\t')
    plot_importance_barplot(
        df_meta_imp, x_col='importance', y_col='feature',
        title="Sample Metadata Features Importance",
        output_path=os.path.join(OUTPUT_DIR, "gat_sample_metadata_importance.png"),
        orientation='vertical'
    )

    df_gat_sal = pd.read_csv(os.path.join(OUTPUT_DIR, "gat_gene_saliency.tsv"), sep='\t')
    plot_importance_barplot(
        df_gat_sal, x_col='saliency_score', y_col='gene_name',
        title="Top GAT Gene Saliency Scores",
        output_path=os.path.join(OUTPUT_DIR, "gat_top_gene_saliency.png"),
        top_n=20
    )

    top_5_saliency = df_gat_sal.head(5)
    top_5_genes_dict = dict(zip(top_5_saliency['gene_id'], top_5_saliency['gene_name']))
    ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}

    records = collect_top_genes_node_features(
        test_loader, top_5_genes_dict.keys(), ensg_to_idx, num_unique_genes, is_mil=False
    )

    plot_gene_feature_boxplots(
        records, top_5_genes_dict,
        class_names=sample_class_names,
        output_dir=OUTPUT_DIR,
        prefix="gat"
    )

    logging.info("GAT Sample Explainability Analysis and Plotting Complete!\n")


if __name__ == "__main__":
    main()