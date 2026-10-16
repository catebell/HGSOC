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
from PatientBagDataset import PatientBagDataset
from models.GAT import GAT
from models.PatientMIL import PatientMILModel
from train_eval_utils import pre_train_imputation_encoding, print_details
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from models.MultiOmicGATSurvival import MultiOmicGATSurvival, CoxPHLoss
from train_functions import (
    train_epoch, evaluate, train_epoch_survival, evaluate_survival,
    train_epoch_mil, evaluate_mil, save_learning_curves
)

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_search_execution.log', mode='w'),
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
use_clinical = config_file_tree.USE_CLINICAL_DATA

task = config_file_tree.TASK
sample_target_column = config_file_tree.SAMPLE_TARGET_COL
patient_target_col = config_file_tree.PATIENT_TARGET_COL

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
path_to_clinical = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")
path_train_val_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_train_val.csv")
path_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR
pretrained_sample_encoder_path = config_file_tree.PRETRAINED_SAMPLE_ENCODER

samples_level_train = config_file_tree.SAMPLES_LEVEL_TRAIN
patients_level_train = config_file_tree.PATIENTS_LEVEL_TRAIN

output_dir = os.path.join("grid_search_advanced")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# HYPERPARAMETER GRID DEFINITION
# -----------------------------------------------------------------------------
params_grid_architecture = {  # TODO check hidden_channels = 128, heads = 4, dropout = 0.2 || hidden_channels = 128, heads = 2, dropout = 0.3
    'lr': [0.001],
    'hidden_channels': [64, 128],  # 32 (less expressivity)
    'heads': [2, 4],  # TODO 2 (captures less relationships but maybe is ok)
    'dropout': [0.2, 0.3],  # 0.5 (too aggressive)
    'batch_size': [16],
    'weight_decay': [1e-4]
}


params_grid_optimization = {
    'lr': [0.001, 0.0005],
    'hidden_channels': [64],
    'heads': [2],
    'dropout': [0.5],
    'batch_size': [16, 32],
    'weight_decay': [1e-4, 1e-3]
}

keys, values = zip(*params_grid_architecture.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
logging.info(f"Total combinations for Grid Search testing: {len(combinations)}\n")

epochs = 100

# -----------------------------------------------------------------------------
# METADATA & CLINICAL DATA LOADING
# -----------------------------------------------------------------------------
df_metadata = pd.read_csv(path_to_metadata, sep="\t")
metadata_train_val_samples = pd.read_csv(path_train_val_csv)

df_metadata_train_val = df_metadata[
    df_metadata['sample_id'].isin(metadata_train_val_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

if patients_level_train:
    cols_to_keep = ['patient_id', patient_target_col]
    if use_clinical:
        cols_to_keep += config_file_tree.CLINICAL_FEATURES
    df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
    clinical_patient_id_col = df_clinical.columns[0]

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

if patients_level_train:
    df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).copy()
    label_encoder_patient = sklearn.preprocessing.LabelEncoder()
    df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])
    num_classes_patient_target = len(label_encoder_patient.classes_)
    logging.info(f"Patient classes identified for MIL ({patient_target_col}): {label_encoder_patient.classes_}\n")
    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

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
logging.info(f"Number of node features used: {num_genes_features}")

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

# Clinical Preprocessing (MIL)
patient_clinical_map = {}
num_patients_features = 0
if patients_level_train and use_clinical:
    train_patients_ids = set(df_metadata_train_val.iloc[train_idx]['patient_id'].unique())
    val_patients_ids = set(df_metadata_train_val.iloc[val_idx]['patient_id'].unique())

    df_clinical_train_raw = df_clinical_clean[
        df_clinical_clean[clinical_patient_id_col].isin(train_patients_ids)].copy()
    df_clinical_val_raw = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(val_patients_ids)].copy()

    df_clinical_train_proc, p_encoders, p_scalers, p_imputers, p_categories = pre_train_imputation_encoding(
        df=df_clinical_train_raw, task='classification', target_col=patient_target_col, is_train=True
    )
    df_clinical_val_proc = pre_train_imputation_encoding(
        df=df_clinical_val_raw, task='classification', target_col=patient_target_col, is_train=False,
        encoders=p_encoders, scalers=p_scalers, imputers=p_imputers, categories=p_categories
    )

    df_clinical_all_proc = pd.DataFrame(pd.concat([df_clinical_train_proc, df_clinical_val_proc], axis=0)).set_index(
        clinical_patient_id_col)
    exclude_patients_cols = {patient_target_col, 'patient_id'}
    patients_feature_cols = [c for c in df_clinical_all_proc.columns if c not in exclude_patients_cols]
    num_patients_features = len(patients_feature_cols)

    for patient_id in df_clinical_all_proc.index:
        feats = df_clinical_all_proc.loc[patient_id, patients_feature_cols].values.astype(np.float32)
        patient_clinical_map[patient_id] = torch.tensor(feats, dtype=torch.float32)

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

if patients_level_train:
    patient_train_samples = [train_val_dataset[i] for i in train_idx]
    patient_train_dataset = PatientBagDataset(patient_train_samples, patient_target_map,
                                              patient_clinical_map=patient_clinical_map if use_clinical else None)
    train_patients_targets = torch.tensor([patient_target_map[pid] for pid in patient_train_dataset.patient_ids],
                                          dtype=torch.long)
    class_counts_patients = torch.clamp(torch.bincount(train_patients_targets, minlength=num_classes_patient_target),
                                        min=1)
    class_weights_patients = (
                len(train_patients_targets) / (len(class_counts_patients) * class_counts_patients.float())).to(device)
    criterion_patients = torch.nn.CrossEntropyLoss(weight=class_weights_patients)

# Subsets & Datasets
train_dataset = Subset(train_val_dataset, train_idx)
val_dataset = Subset(train_val_dataset, val_idx)

if patients_level_train:
    patient_val_dataset = PatientBagDataset([train_val_dataset[i] for i in val_idx], patient_target_map,
                                            patient_clinical_map=patient_clinical_map if use_clinical else None)
    patients_train_loader = DataLoader(patient_train_dataset, batch_size=1, shuffle=True)
    patients_val_loader = DataLoader(patient_val_dataset, batch_size=1, shuffle=False)

# -----------------------------------------------------------------------------
# GRID SEARCH LOOP
# -----------------------------------------------------------------------------
grid_results = []

for i, params in enumerate(combinations):
    logging.info(f"========== GRIDSEARCH TEST {i + 1}/{len(combinations)} | CONFIG: {params} ==========")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    best_gat_weights = None
    gat_metrics = {}

    # 1. SAMPLES LEVEL TRAINING (GAT)

    if task == 'classification':
        model_samples_encoder = GAT(
            num_node_features=num_genes_features,
            num_classes=num_classes_samples,
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

    print_details(model_samples_encoder)

    if samples_level_train:
        optimizer_samples = torch.optim.AdamW(model_samples_encoder.parameters(), lr=params['lr'],
                                              weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_samples,
                                                               mode='min' if task == 'classification' else 'max',
                                                               factor=0.5, patience=7)

        best_val_score = float('inf') if task == 'classification' else -float('inf')
        early_stopping_counter = 0

        # Learning curves tracking for GAT
        history_gat = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

        for epoch in range(1, epochs + 1):
            if task == 'classification':
                train_loss, train_acc = train_epoch(device, model_samples_encoder, train_loader, optimizer_samples,
                                                    criterion_samples)
                val_loss, val_metrics = evaluate(device, model_samples_encoder, val_loader, criterion_samples)
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
                train_loss = train_epoch_survival(device, model_samples_encoder, train_loader, optimizer_samples,
                                                  criterion_samples)
                val_loss, val_metrics = evaluate_survival(device, model_samples_encoder, val_loader, criterion_samples)
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
            fold_num=i+1,
            output_dir=os.path.join(output_dir, 'learning_curves_grid_search', f'GAT_grid_cfg{i + 1}'),
            is_grid_search=True
        )

        logging.info(f"[GAT COMPLETED] Config {i + 1} | Best Val Score: {best_val_score:.4f} | Metrics: {gat_metrics}\n")

    else:
        # Load pre-trained encoder if SAMPLES_LEVEL_TRAIN is False
        logging.info(f"Loading pre-trained Sample Encoder from: {pretrained_sample_encoder_path}")
        model_samples_encoder.load_state_dict(torch.load(pretrained_sample_encoder_path, map_location=device))


    # 2. PATIENTS LEVEL TRAINING (MIL)

    mil_metrics = {}
    if patients_level_train:
        if best_gat_weights is not None:
            model_samples_encoder.load_state_dict(best_gat_weights)

        model_patient_encoder = PatientMILModel(
            sample_encoder=model_samples_encoder,
            num_classes=num_classes_patient_target,
            num_clinical_features=num_patients_features if use_clinical else 0
        ).to(device)

        optimizer_patients = torch.optim.AdamW(
            [
                {'params': model_patient_encoder.sample_encoder.parameters(), 'lr': params['lr'] * 0.01},
                {'params': model_patient_encoder.pooling.parameters(), 'lr': params['lr']},
                {'params': model_patient_encoder.classifier.parameters(), 'lr': params['lr']},
            ],
            weight_decay=params['weight_decay'],
        )

        best_val_loss_mil = float('inf')
        early_stopping_counter = 0

        # Learning curves tracking for MIL
        history_mil = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch_mil(device, model_patient_encoder, patients_train_loader,
                                                    optimizer_patients, criterion_patients)
            val_loss, val_m = evaluate_mil(device, model_patient_encoder, patients_val_loader, criterion_patients)

            history_mil['train_loss'].append(train_loss)
            history_mil['val_loss'].append(val_loss)
            history_mil['train_metric'].append(train_acc)
            history_mil['val_metric'].append(val_m['acc'])

            if val_loss < best_val_loss_mil:
                best_val_loss_mil = val_loss
                mil_metrics = val_m
                mil_metrics['val_loss'] = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter > 15:
                break

        # Save Learning Curves plot for current MIL configuration
        save_learning_curves(
            history_mil,
            fold_num=i+1,
            output_dir=os.path.join('learning_curves_grid_search', f'MIL_grid_cfg{i + 1}'),
            is_grid_search=True
        )

        logging.info(
            f"[MIL COMPLETED] Config {i + 1} | Best Val Loss: {best_val_loss_mil:.4f} | Metrics: {mil_metrics}\n")

    # Record current configuration metrics
    record = {**params}
    if samples_level_train:
        for k, v in gat_metrics.items():
            record[f"gat_{k}"] = v
    if patients_level_train:
        for k, v in mil_metrics.items():
            record[f"mil_{k}"] = v

    grid_results.append(record)

    # Clean Memory
    del model_samples_encoder
    if patients_level_train:
        del model_patient_encoder
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# HEATMAP GENERATION FUNCTION
# -----------------------------------------------------------------------------
def plot_grid_search_heatmaps(df_res, out_dir):
    """
    Generates and saves visual heatmaps for quick comparison of Grid Search results.
    """
    if df_res.empty:
        return

    metric_cols = [c for c in df_res.columns if c.startswith('gat_') or c.startswith('mil_')]
    if not metric_cols:
        return

    # 1. Summary Heatmap: Configurations vs Evaluated Metrics
    plot_df = df_res.copy()
    plot_df['Config'] = [f"Cfg {i + 1}" for i in range(len(plot_df))]
    plot_df = plot_df.set_index('Config')[metric_cols]

    plt.figure(figsize=(max(8, len(metric_cols) * 1.8), max(6, len(plot_df) * 0.35)))
    sns.heatmap(plot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("Grid Search Metrics Summary per Configuration", fontsize=14, pad=12)
    plt.ylabel("Configuration Index")
    plt.xlabel("Evaluated Metrics")
    plt.tight_layout()
    heatmap_summary_path = os.path.join(out_dir, "grid_search_summary_heatmap.png")
    plt.savefig(heatmap_summary_path, dpi=300)
    plt.close()
    logging.info(f"Summary heatmap saved to: {heatmap_summary_path}")

    # 2. Pivot Heatmap: Primary Metric across Hidden Channels vs Learning Rate
    target_metric = 'mil_auc' if 'mil_auc' in df_res.columns else (
        'gat_auc' if 'gat_auc' in df_res.columns else ('gat_c_index' if 'gat_c_index' in df_res.columns else None)
    )

    if target_metric and 'hidden_channels' in df_res.columns and 'lr' in df_res.columns:
        plt.figure(figsize=(8, 6))
        pivot_df = df_res.pivot_table(index='hidden_channels', columns='lr', values=target_metric, aggfunc='mean')
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis", cbar=True, linewidths=0.5)
        plt.title(f"Mean {target_metric.upper()} (Hidden Channels vs Learning Rate)", fontsize=12, pad=12)
        plt.xlabel("Learning Rate (lr)")
        plt.ylabel("Hidden Channels")
        plt.tight_layout()
        pivot_path = os.path.join(out_dir, f"grid_search_pivot_{target_metric}.png")
        plt.savefig(pivot_path, dpi=300)
        plt.close()
        logging.info(f"Pivot heatmap saved to: {pivot_path}")


# -----------------------------------------------------------------------------
# FINAL EXPORT AND LOGGING
# -----------------------------------------------------------------------------
df_results = pd.DataFrame(grid_results)
results_csv_path = os.path.join(output_dir, "grid_search_results.csv")
df_results.to_csv(results_csv_path, index=False)

# Generate Heatmaps
plot_grid_search_heatmaps(df_results, output_dir)

logging.info("==================================================")
logging.info("GRID SEARCH COMPLETED SUCCESSFULLY")
logging.info(f"Results CSV saved to: {results_csv_path}")
logging.info("==================================================\n")
logging.info("\n" + df_results.to_string())