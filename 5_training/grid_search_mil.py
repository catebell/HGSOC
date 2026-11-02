import logging
import os
import itertools
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from PatientBagDataset import PatientBagDataset
from models.GAT import GAT
from models.PatientMIL_Enhanced_ModalityDropout import PatientMILModel
from train_eval_utils import pre_train_imputation_encoding, print_details
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from train_functions import train_epoch_mil, evaluate_mil, save_learning_curves

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler('grid_search_mil.log', mode='w'),
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
# CONFIGURAZIONE E CARICAMENTO ASSET DALLA CARTELLA GAT
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

gat_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
pretrained_sample_encoder_path = os.path.join(
    gat_dir,
    f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt"
)

output_dir = os.path.join("grid_search_mil_results2")
os.makedirs(output_dir, exist_ok=True)

logging.info(f"Pre-trained GAT model path: {pretrained_sample_encoder_path}")
logging.info(f"Loading GAT joblib preprocessors directly from directory: {gat_dir}\n")

# ==============================================================================
# FASE 1: Architettura & Capacità Rappresentazionale MIL (12 combinazioni)
# Obiettivo: Identificare la dimensione ideale degli spazi vettoriali.
# ==============================================================================
params_grid_phase1 = {
    'tissue_emb_dim': [0, 8, 16],  # 0 = disattivo, 8 o 16 = embedding compatto
    'pooling_hidden_dim': [32, 64],  # Dimensione nascosta Attention Pooling
    'classifier_hidden_dim': [32, 64],  # Layer denso classificatore finale

    # Parametri temporaneamente fissati sui valori migliori già emersi:
    'batch_size': [8],
    'subsample_rate': [0.85],
    'dropout': [0.3],
    'sample_encoder_lr': [5e-6],
    'mil_lr': [1e-3],
    'weight_decay': [1e-2],
    'label_smoothing': [0.0]
}

# ==============================================================================
# FASE 2: Regolarizzazione & Augmentation del Bag (18 combinazioni)
# Obiettivo: Fissare l'architettura migliore trovata nella Fase 1 e prevenire l'overfitting.
# ==============================================================================
params_grid_phase2 = {
    'dropout': [0.2, 0.3, 0.4],  # Dropout attention + classificatore
    'subsample_rate': [0.75, 0.85, 1.0],  # Augmentation bag subsampling
    'label_smoothing': [0.0, 0.1],  # Smoothing sulle label della CrossEntropy

    # Eredita 'tissue_emb_dim', 'pooling_hidden_dim', 'classifier_hidden_dim' migliori dalla Fase 1
    'tissue_emb_dim': [16],
    'pooling_hidden_dim': [64],
    'classifier_hidden_dim': [32],

    'batch_size': [8],
    'sample_encoder_lr': [5e-6],
    'mil_lr': [1e-3],
    'weight_decay': [1e-2],
}

# ==============================================================================
# FASE 3: Ottimizzazione & Dinamiche di Training (12 combinazioni)
# Obiettivo: Fissare Architettura (Fase 1) e Regolarizzazione (Fase 2) per ottimizzare i gradienti.
# ==============================================================================
params_grid_phase3 = {
    'sample_encoder_lr': [0.0, 5e-6],  # 0.0 = GAT congelata, 5e-6 = Fine-tuning leggero
    'mil_lr': [3e-4, 1e-3],  # Learning rate MIL
    'weight_decay': [1e-3, 1e-2],  # Regolarizzazione L2
    'batch_size': [4, 8],  # Batch size pazienti

    'tissue_emb_dim': [16],
    'pooling_hidden_dim': [64],
    'classifier_hidden_dim': [32],

    'dropout': [0.3],
    'subsample_rate': [0.85],
    'label_smoothing': [0.0],
}

# Modificare qui la fase da eseguire (es. params_grid_phase1, params_grid_phase2, params_grid_phase3)
keys, values = zip(*params_grid_phase2.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
logging.info(f"Total MIL combinations to test: {len(combinations)}\n")

epochs = 75

# -----------------------------------------------------------------------------
# CARICAMENTO DATI
# -----------------------------------------------------------------------------
df_metadata = pd.read_csv(path_to_metadata, sep="\t")
metadata_train_val_samples = pd.read_csv(path_train_val_csv)

df_metadata_train_val = df_metadata[
    df_metadata['sample_id'].isin(metadata_train_val_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

cols_to_keep = ['patient_id', patient_target_col]
if use_clinical:
    cols_to_keep += config_file_tree.CLINICAL_FEATURES
df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
clinical_patient_id_col = 'patient_id'

df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).copy()
label_encoder_patient = sklearn.preprocessing.LabelEncoder()
df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])
num_classes_patient_target = len(label_encoder_patient.classes_)
patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

# Target tessuto per il Sample Encoder
label_encoder_sample = sklearn.preprocessing.LabelEncoder()
df_metadata_train_val[sample_target_column] = label_encoder_sample.fit_transform(
    df_metadata_train_val[sample_target_column])
num_classes_samples = len(label_encoder_sample.classes_)

# Caricamento Grafi in RAM
logging.info("Pre-loading graphs into RAM...")
raw_train_val_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_train_val, task=task,
                                             target_col=sample_target_column)
train_val_dataset = [raw_train_val_dataset[i] for i in
                     tqdm(range(len(raw_train_val_dataset)), desc="Loading Train/Val")]

for graph in train_val_dataset:
    graph.x_raw = graph.x.clone()

num_genes_features = train_val_dataset[0].x.shape[1]

# -----------------------------------------------------------------------------
# CARICAMENTO PRE-PROCESSORI JOBLIB DALLA CARTELLA GAT
# -----------------------------------------------------------------------------

# 1. Scaler di Nodo GAT
node_scaler_file = os.path.join(gat_dir, f"node_scaler_{sample_target_column}.joblib")
if os.path.exists(node_scaler_file):
    node_scaler = joblib.load(node_scaler_file)
    logging.info(f"Loaded GAT Node Scaler from: {node_scaler_file}")
    for graph in train_val_dataset:
        scaled_x = node_scaler.transform(graph.x_raw.numpy())
        graph.x = torch.tensor(scaled_x, dtype=torch.float32)
else:
    logging.warning(f"Node scaler not found in {gat_dir}. Fitting in-fold StandardScaler.")

# 2. Pre-processori Metadati Campione GAT
num_sample_features = 0
if use_metadata:
    metadata_prep_file = os.path.join(gat_dir, f"metadata_preprocessors_{sample_target_column}.joblib")
    if os.path.exists(metadata_prep_file):
        meta_prep = joblib.load(metadata_prep_file)
        logging.info(f"Loaded GAT Metadata Preprocessors from: {metadata_prep_file}")
        df_meta_proc = pre_train_imputation_encoding(
            df=df_metadata_train_val, task=task, target_col=sample_target_column, is_train=False,
            encoders=meta_prep['encoders'], scalers=meta_prep['scalers'],
            imputers=meta_prep['imputers'], categories=meta_prep['categories']
        ).set_index('sample_id')

        exclude_cols = {sample_target_column, 'patient_id'}
        sample_feature_cols = [c for c in df_meta_proc.columns if c not in exclude_cols]
        num_sample_features = len(sample_feature_cols)
        update_graphs_metadata(train_val_dataset, df_meta_proc, sample_feature_cols)

# -----------------------------------------------------------------------------
# STRATIFIED GROUP K-FOLD SETUP PER IL GRID SEARCH
# -----------------------------------------------------------------------------
groups = df_metadata_train_val['patient_id'].values
patient_to_clinical_target = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()
composite_y = [f"{row[sample_target_column]}_{patient_to_clinical_target[row['patient_id']]}" for _, row in
               df_metadata_train_val.iterrows()]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)

# Pre-costruzione split 5-fold per riutilizzarli identici per ogni configurazione
cv_splits = list(sgkf.split(train_val_dataset, composite_y, groups=groups))

# -----------------------------------------------------------------------------
# GRID SEARCH LOOP (Esecuzione su 5-Fold Cross Validation per ogni Cfg)
# -----------------------------------------------------------------------------
grid_summary_results = []

for cfg_idx, params in enumerate(combinations):
    logging.info(f"==================================================================")
    logging.info(f"GRID SEARCH MIL CONFIG {cfg_idx + 1}/{len(combinations)} | {params}")
    logging.info(f"==================================================================")

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits):

        # Preprocessing Dati Clinici In-Fold
        patient_clinical_map = {}
        num_patients_features = 0
        if use_clinical:
            train_pids = set(df_metadata_train_val.iloc[train_idx]['patient_id'].unique())
            val_pids = set(df_metadata_train_val.iloc[val_idx]['patient_id'].unique())

            df_clin_train = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(train_pids)].copy()
            df_clin_val = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(val_pids)].copy()

            df_clin_train_proc, p_enc, p_scale, p_imp, p_cat = pre_train_imputation_encoding(
                df=df_clin_train, task='classification', target_col=patient_target_col, is_train=True
            )
            df_clin_val_proc = pre_train_imputation_encoding(
                df=df_clin_val, task='classification', target_col=patient_target_col, is_train=False,
                encoders=p_enc, scalers=p_scale, imputers=p_imp, categories=p_cat
            )

            df_clin_all = pd.DataFrame(pd.concat([df_clin_train_proc, df_clin_val_proc], axis=0)).set_index(
                clinical_patient_id_col)
            exclude_pcols = {patient_target_col, 'patient_id'}
            p_feature_cols = [c for c in df_clin_all.columns if c not in exclude_pcols]
            num_patients_features = len(p_feature_cols)

            for pid in df_clin_all.index:
                feats = df_clin_all.loc[pid, p_feature_cols].values.astype(np.float32)
                patient_clinical_map[pid] = torch.tensor(feats, dtype=torch.float32)

        # Creazione Bag Dataset
        train_samples = [train_val_dataset[i] for i in train_idx]
        val_samples = [train_val_dataset[i] for i in val_idx]

        patient_train_dataset = PatientBagDataset(train_samples, patient_target_map,
                                                  patient_clinical_map=patient_clinical_map if use_clinical else None)
        patient_val_dataset = PatientBagDataset(val_samples, patient_target_map,
                                                patient_clinical_map=patient_clinical_map if use_clinical else None)

        patients_train_loader = DataLoader(patient_train_dataset, batch_size=params['batch_size'], shuffle=True)
        patients_val_loader = DataLoader(patient_val_dataset, batch_size=params['batch_size'], shuffle=False)

        # Pesatura Classi Paziente
        train_p_targets = torch.tensor([patient_target_map[pid] for pid in patient_train_dataset.patient_ids],
                                       dtype=torch.long)
        class_counts = torch.bincount(train_p_targets, minlength=num_classes_patient_target).clamp(min=1)
        class_weights = (len(train_p_targets) / (len(class_counts) * class_counts.float())).to(device)
        criterion_patients = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=params['label_smoothing'])

        # Inizializzazione Modello GAT ed Enhanced MIL Model
        num_nodes_per_graph = train_val_dataset[0].x.shape[0]
        model_samples_encoder = GAT(
            num_node_features=num_genes_features,
            num_classes=num_classes_samples,
            num_nodes=num_nodes_per_graph,
            num_sample_features=num_sample_features if use_metadata else 0,
        ).to(device)

        model_samples_encoder.load_state_dict(torch.load(pretrained_sample_encoder_path, map_location=device))

        model_patient_encoder = PatientMILModel(
            sample_encoder=model_samples_encoder,
            num_node_features=num_genes_features,
            num_classes=num_classes_patient_target,
            num_tissue_classes=num_classes_samples,
            tissue_emb_dim=params['tissue_emb_dim'],
            pooling_hidden_dim=params['pooling_hidden_dim'],
            classifier_hidden_dim=params['classifier_hidden_dim'],
            num_clinical_features=num_patients_features if use_clinical else 0,
            dropout=params['dropout'],
            subsample_rate=params['subsample_rate']
        ).to(device)

        # Configurazione Parametri & Learning Rate Differenziato (Completo di tutte le componenti Enhanced MIL)
        if params['sample_encoder_lr'] == 0.0:
            for p in model_patient_encoder.sample_encoder.parameters():
                p.requires_grad = False
        else:
            for p in model_patient_encoder.sample_encoder.parameters():
                p.requires_grad = True

        sample_encoder_params = set(model_patient_encoder.sample_encoder.parameters())
        mil_head_params = [p for p in model_patient_encoder.parameters() if p not in sample_encoder_params]

        opt_params = []
        if params['sample_encoder_lr'] > 0.0:
            opt_params.append({
                'params': list(sample_encoder_params),
                'lr': params['sample_encoder_lr'],
                'weight_decay': params['weight_decay']
            })

        opt_params.append({
            'params': mil_head_params,
            'lr': params['mil_lr'],
            'weight_decay': params['weight_decay']
        })

        optimizer_patients = torch.optim.AdamW(opt_params)
        scheduler_patients = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_patients, T_max=epochs, eta_min=1e-6)

        # Epoch Loop MIL Fine-Tuning
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_fold_metrics = {}
        history_mil = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch_mil(device, model_patient_encoder, patients_train_loader,
                                                    optimizer_patients, criterion_patients)
            val_loss, val_m = evaluate_mil(device, model_patient_encoder, patients_val_loader, criterion_patients)

            scheduler_patients.step()

            history_mil['train_loss'].append(train_loss)
            history_mil['val_loss'].append(val_loss)
            history_mil['train_metric'].append(train_acc)
            history_mil['val_metric'].append(val_m['acc'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_fold_metrics = val_m
                best_fold_metrics['val_loss'] = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter > 10:
                break

        fold_metrics.append(best_fold_metrics)

        # Salva la curva d'apprendimento del primo fold di ogni configurazione
        if fold == 0:
            save_learning_curves(
                history_mil,
                fold_num=cfg_idx + 1,
                output_dir=os.path.join(output_dir, "learning_curves_grid"),
                is_grid_search=True
            )

        del model_samples_encoder, model_patient_encoder
        torch.cuda.empty_cache()

    # Calcolo medie e deviazioni standard sulle 5 fold per la configurazione corrente
    df_fold_m = pd.DataFrame(fold_metrics)
    record = {**params}
    record['mean_val_loss'] = df_fold_m['val_loss'].mean()
    record['std_val_loss'] = df_fold_m['val_loss'].std()
    record['mean_val_acc'] = df_fold_m['acc'].mean()
    record['std_val_acc'] = df_fold_m['acc'].std()
    record['mean_val_f1'] = df_fold_m['f1'].mean()
    record['std_val_f1'] = df_fold_m['f1'].std()
    record['mean_val_auc'] = df_fold_m['auc'].mean()
    record['std_val_auc'] = df_fold_m['auc'].std()

    grid_summary_results.append(record)
    logging.info(
        f"[CFG {cfg_idx + 1} COMPLETED 5-FOLD CV] Loss: {record['mean_val_loss']:.4f} ± {record['std_val_loss']:.4f} | AUC: {record['mean_val_auc']:.4f} ± {record['std_val_auc']:.4f} | Acc: {record['mean_val_acc']:.4f}")

# -----------------------------------------------------------------------------
# ESPORTAZIONE E GENERAZIONE HEATMAP
# -----------------------------------------------------------------------------
df_summary = pd.DataFrame(grid_summary_results)
df_summary = df_summary.sort_values(by='mean_val_auc', ascending=False)
summary_csv_path = os.path.join(output_dir, "grid_search_mil_summary.csv")
df_summary.to_csv(summary_csv_path, index=False)

# Heatmap riassuntiva
plt.figure(figsize=(10, max(6, len(df_summary) * 0.4)))
plot_df = df_summary.set_index(
    df_summary.apply(lambda r: f"Cfg {r.name + 1}: encoder_lr={r['sample_encoder_lr']}, emb={r['tissue_emb_dim']}",
                     axis=1))[['mean_val_loss', 'mean_val_acc', 'mean_val_f1', 'mean_val_auc']]
sns.heatmap(plot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Grid Search MIL: Mean 5-Fold Metrics per Configuration", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grid_search_mil_summary_heatmap.png"), dpi=300)
plt.close()

logging.info("==================================================")
logging.info("GRID SEARCH MIL COMPLETED SUCCESSFULLY")
logging.info(f"Summary CSV saved to: {summary_csv_path}")
logging.info("==================================================\n")
logging.info("" + df_summary[
    ['sample_encoder_lr', 'mil_lr', 'weight_decay', 'tissue_emb_dim', 'subsample_rate', 'mean_val_loss', 'mean_val_auc',
     'mean_val_acc']].to_string())