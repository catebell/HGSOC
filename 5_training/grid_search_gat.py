import logging
import os
import itertools
import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from models.GAT import GAT
from train_eval_utils import pre_train_imputation_encoding, print_details
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from models.MultiOmicGATSurvival import MultiOmicGATSurvival, CoxPHLoss
from train_functions import (
    train_epoch, evaluate, train_epoch_survival, evaluate_survival,
    save_learning_curves
)

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler('grid_search_gat_execution.log', mode='w'),
        logging.StreamHandler()
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    cpus_allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    try:
        torch.set_num_threads(cpus_allocated)
        torch.set_num_interop_threads(max(1, cpus_allocated // 2))
    except RuntimeError as e:
        logging.warning(f"Failed setting interop_threads: {e}")
    logging.info(f"Allocated threads for PyTorch: {torch.get_num_threads()}")

logging.info(f"Device: {device}\n")

# -----------------------------------------------------------------------------
# GENERAL PARAMETERS & CONFIGURATION
# -----------------------------------------------------------------------------
use_metadata = config_file_tree.USE_SAMPLES_METADATA
task = config_file_tree.TASK
sample_target_column = config_file_tree.SAMPLE_TARGET_COL

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
path_train_val_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_train_val.csv")
path_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR

output_dir = os.path.join("grid_search_gat_results")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# HYPERPARAMETER SUBSETS / PHASES (DEFAULT VALUES DERIVED FROM KFOLD & GAT_2)
# -----------------------------------------------------------------------------
# Defaults: gene_emb_dim=8, hidden_channels=64, heads=2, dropout=0.3, lr=0.001, weight_decay=1e-4, batch_size=16

# FASE 1: Architettura GAT & Spazio Vettoriale Latente (18 combinazioni)
params_grid_phase1 = {
    'gene_emb_dim': [4, 8, 16],        # Dimensione Embedding Identità Genica
    'hidden_channels': [32, 64, 128],   # Canali nascosti per GATv2Conv
    'heads': [1, 2],                   # Teste di attenzione
    'dropout': [0.3],                  # Default da GAT / K-Fold
    'lr': [0.001],                     # Default da K-Fold
    'weight_decay': [1e-4],            # Default da K-Fold
    'batch_size': [16]                 # Default da K-Fold
}

# FASE 2: Regolarizzazione & Prevenzione Overfitting (20 combinazioni)
params_grid_phase2 = {
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],     # Dropout su strati convoluzionali e readout
    'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2], # Weight Decay (AdamW)
    'gene_emb_dim': [8],                      # Default / Ottimale Fase 1
    'hidden_channels': [64],                  # Default / Ottimale Fase 1
    'heads': [2],                             # Default / Ottimale Fase 1
    'lr': [0.001],
    'batch_size': [16]
}

# FASE 3: Dinamiche di Ottimizzazione & Batching (12 combinazioni)
params_grid_phase3 = {
    'lr': [0.0003, 0.0005, 0.001, 0.003],     # Learning rate di AdamW
    'batch_size': [8, 16, 32],                # Dimensione del mini-batch
    'gene_emb_dim': [8],
    'hidden_channels': [64],
    'heads': [2],
    'dropout': [0.3],
    'weight_decay': [1e-4]
}

# SELEZIONARE LA FASE DA ESEGUIRE (CAMBIARE QUI PER ESEGUIRE LE DIVERSE FASI)
selected_grid = params_grid_phase1

keys, values = zip(*selected_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
logging.info(f"Total GAT combinations to test in selected grid: {len(combinations)}\n")

epochs = 75

# -----------------------------------------------------------------------------
# METADATA & DATA LOADING
# -----------------------------------------------------------------------------
df_metadata = pd.read_csv(path_to_metadata, sep="\t")
metadata_train_val_samples = pd.read_csv(path_train_val_csv)

df_metadata_train_val = df_metadata[
    df_metadata['sample_id'].isin(metadata_train_val_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

# -----------------------------------------------------------------------------
# TARGET ENCODING
# -----------------------------------------------------------------------------
if task == 'classification':
    label_encoder_sample = sklearn.preprocessing.LabelEncoder()
    df_metadata_train_val[sample_target_column] = label_encoder_sample.fit_transform(
        df_metadata_train_val[sample_target_column]
    )
    logging.info(f"Sample classes identified ({sample_target_column}): {label_encoder_sample.classes_}")
else:
    logging.info(f"Starting Grid Search on survival task using target: {sample_target_column}\n")

# -----------------------------------------------------------------------------
# LOADING GRAPHS INTO RAM
# -----------------------------------------------------------------------------
logging.info("Loading Train/Val dataset...")
raw_train_val_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_train_val, task=task,
                                             target_col=sample_target_column)

logging.info("Pre-loading graphs into RAM...")
train_val_dataset = [raw_train_val_dataset[i] for i in
                     tqdm(range(len(raw_train_val_dataset)), desc="Loading Train/Val")]

for graph in train_val_dataset:
    graph.x_raw = graph.x.clone()

num_genes_features = train_val_dataset[0].x.shape[1]
num_nodes_per_graph = train_val_dataset[0].x.shape[0]
logging.info(f"Number of node features used: {num_genes_features}")
logging.info(f"Number of nodes per graph: {num_nodes_per_graph}")

# -----------------------------------------------------------------------------
# TRAIN / VAL SPLIT CREATION FOR GRID SEARCH (Grouped by Patient)
# -----------------------------------------------------------------------------
groups = df_metadata_train_val['patient_id'].values
if task == 'classification':
    y_skf = torch.tensor(df_metadata_train_val[sample_target_column].values, dtype=torch.long)
else:
    y_skf = torch.tensor(df_metadata_train_val[sample_target_column].values, dtype=torch.float32)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
train_idx, val_idx = next(sgkf.split(train_val_dataset, y_skf, groups=groups))

# Scaling node features in-fold
train_nodes_np = np.vstack([train_val_dataset[i].x_raw.numpy() for i in train_idx])
node_scaler = StandardScaler()
node_scaler.fit(train_nodes_np)

for i in train_idx:
    train_val_dataset[i].x = torch.tensor(node_scaler.transform(train_val_dataset[i].x_raw.numpy()),
                                          dtype=torch.float32)
for i in val_idx:
    train_val_dataset[i].x = torch.tensor(node_scaler.transform(train_val_dataset[i].x_raw.numpy()),
                                          dtype=torch.float32)

# Metadata Preprocessing
num_sample_features = 0
if use_metadata:
    df_metadata_train_raw = df_metadata_train_val.iloc[train_idx].copy()
    df_metadata_val_raw = df_metadata_train_val.iloc[val_idx].copy()

    df_train_proc, encoders, scalers, imputers, categories = pre_train_imputation_encoding(
        df=df_metadata_train_raw, task=task, target_col=sample_target_column, is_train=True
    )
    df_val_proc = pre_train_imputation_encoding(
        df=df_metadata_val_raw, task=task, target_col=sample_target_column, is_train=False,
        encoders=encoders, scalers=scalers, imputers=imputers, categories=categories
    )

    df_train_val_proc = pd.DataFrame(pd.concat([df_train_proc, df_val_proc], axis=0)).set_index('sample_id')
    exclude_cols = {sample_target_column, 'patient_id'}
    sample_feature_cols = [c for c in df_train_val_proc.columns if c not in exclude_cols]
    num_sample_features = len(sample_feature_cols)

    update_graphs_metadata(train_val_dataset, df_train_val_proc, sample_feature_cols)

# Loss Criterions
if task == 'classification':
    y_labels_samples = torch.tensor(df_metadata_train_val.iloc[train_idx][sample_target_column].values,
                                    dtype=torch.long)
    class_counts_samples = torch.bincount(y_labels_samples)
    class_weights_samples = len(y_labels_samples) / (len(class_counts_samples) * class_counts_samples.float()).to(
        device)
    criterion_samples = torch.nn.CrossEntropyLoss(weight=class_weights_samples)
    num_classes_samples = len(torch.unique(y_skf))
elif task == 'survival':
    criterion_samples = CoxPHLoss()

# Subsets & Datasets
train_dataset = Subset(train_val_dataset, train_idx)
val_dataset = Subset(train_val_dataset, val_idx)

# -----------------------------------------------------------------------------
# GRID SEARCH LOOP (GAT ONLY)
# -----------------------------------------------------------------------------
grid_results = []

for i, params in enumerate(combinations):
    logging.info(f"========== GRIDSEARCH TEST {i + 1}/{len(combinations)} | CONFIG: {params} ==========")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    best_gat_weights = None
    gat_metrics = {}

    if task == 'classification':
        model_samples_encoder = GAT(
            num_node_features=num_genes_features,
            num_classes=num_classes_samples,
            num_nodes=num_nodes_per_graph,
            gene_emb_dim=params['gene_emb_dim'],
            hidden_channels=params['hidden_channels'],
            heads=params['heads'],
            dropout=params['dropout'],
            num_sample_features=num_sample_features if use_metadata else 0,
        ).to(device)

    elif task == 'survival':
        model_samples_encoder = MultiOmicGATSurvival(
            in_node_features=num_genes_features,
            hidden_dim=params['hidden_channels'],
            out_channels=1,
            heads=params['heads'],
            dropout_rate=params['dropout'],
            num_samples_features=num_sample_features if use_metadata else 0,
        ).to(device)

    logging.info(str(model_samples_encoder) + "\n")

    optimizer_samples = torch.optim.AdamW(
        model_samples_encoder.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_samples,
        mode='min' if task == 'classification' else 'max',
        factor=0.5,
        patience=7
    )

    best_val_score = float('inf') if task == 'classification' else -float('inf')
    early_stopping_counter = 0

    history_gat = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

    for epoch in range(1, epochs + 1):
        if task == 'classification':
            train_loss, train_acc = train_epoch(
                device, model_samples_encoder, train_loader, optimizer_samples, criterion_samples
            )
            val_loss, val_metrics = evaluate(
                device, model_samples_encoder, val_loader, criterion_samples
            )
            scheduler.step(val_loss)

            history_gat['train_loss'].append(train_loss)
            history_gat['val_loss'].append(val_loss)
            history_gat['train_metric'].append(train_acc)
            history_gat['val_metric'].append(val_metrics['acc'])

            if val_loss < best_val_score:
                best_val_score = val_loss
                gat_metrics = val_metrics
                gat_metrics['val_loss'] = val_loss
                best_gat_weights = {k: v.cpu().clone() for k, v in model_samples_encoder.state_dict().items()}
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        elif task == 'survival':
            train_loss = train_epoch_survival(
                device, model_samples_encoder, train_loader, optimizer_samples, criterion_samples
            )
            val_loss, val_metrics = evaluate_survival(
                device, model_samples_encoder, val_loader, criterion_samples
            )
            val_c_index = val_metrics['c_index']
            scheduler.step(val_c_index)

            history_gat['train_loss'].append(train_loss)
            history_gat['val_loss'].append(val_loss)
            history_gat['train_metric'].append(0.0)
            history_gat['val_metric'].append(val_c_index)

            if val_c_index > best_val_score:
                best_val_score = val_c_index
                gat_metrics = val_metrics
                gat_metrics['val_loss'] = val_loss
                best_gat_weights = {k: v.cpu().clone() for k, v in model_samples_encoder.state_dict().items()}
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        if early_stopping_counter > 15:
            break

    # Save Learning Curves plot for current GAT configuration
    save_learning_curves(
        history_gat,
        fold_num=i + 1,
        output_dir=os.path.join(output_dir, 'learning_curves_grid_search', f'GAT_grid_cfg{i + 1}'),
        is_grid_search=True
    )

    logging.info(f"[GAT COMPLETED] Config {i + 1} | Best Val Score: {best_val_score:.4f} | Metrics: {gat_metrics}\n")

    # Record metrics
    record = {**params}
    for k, v in gat_metrics.items():
        record[f"gat_{k}"] = v

    grid_results.append(record)

    del model_samples_encoder
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# HEATMAP GENERATION FUNCTION
# -----------------------------------------------------------------------------
def plot_grid_search_heatmaps(df_res, out_dir):
    """
    Generates and saves visual heatmaps for GAT Grid Search results.
    """
    if df_res.empty:
        return

    metric_cols = [c for c in df_res.columns if c.startswith('gat_')]
    if not metric_cols:
        return

    plot_df = df_res.copy()
    plot_df['Config'] = [f"Cfg {i + 1}" for i in range(len(plot_df))]
    plot_df = plot_df.set_index('Config')[metric_cols]

    plt.figure(figsize=(max(8, len(metric_cols) * 1.8), max(6, len(plot_df) * 0.35)))
    sns.heatmap(plot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("GAT Grid Search Metrics Summary per Configuration", fontsize=14, pad=12)
    plt.ylabel("Configuration Index")
    plt.xlabel("Evaluated Metrics")
    plt.tight_layout()
    heatmap_summary_path = os.path.join(out_dir, "grid_search_gat_summary_heatmap.png")
    plt.savefig(heatmap_summary_path, dpi=300)
    plt.close()
    logging.info(f"Summary heatmap saved to: {heatmap_summary_path}")


# -----------------------------------------------------------------------------
# FINAL EXPORT AND LOGGING
# -----------------------------------------------------------------------------
df_results = pd.DataFrame(grid_results)
results_csv_path = os.path.join(output_dir, "grid_search_gat_results.csv")
df_results.to_csv(results_csv_path, index=False)

plot_grid_search_heatmaps(df_results, output_dir)

logging.info("==================================================")
logging.info("GAT GRID SEARCH COMPLETED SUCCESSFULLY")
logging.info(f"Results CSV saved to: {results_csv_path}")
logging.info("==================================================\n")
logging.info("\n" + df_results.to_string())