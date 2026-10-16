import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from PatientBagDataset import PatientBagDataset
from models.GAT import GAT
from models.PatientMIL import PatientMILModel
from train_eval_utils import print_details, pre_train_imputation_encoding
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from models.MultiOmicGATSurvival import CoxPHLoss
from train_functions import train_epoch, evaluate, train_epoch_survival, evaluate_survival, show_fold_results, \
    train_epoch_mil, evaluate_mil, save_learning_curves

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log', mode='w'),
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
        logging.warning(f"Impossibile impostare interop_threads: {e}")

    logging.info(f"Allocated threads for PyTorch: {torch.get_num_threads()}")

logging.info(f"Device: {device}\n")


""""" set general parameters """""

use_metadata = config_file_tree.USE_SAMPLES_METADATA
use_clinical = config_file_tree.USE_CLINICAL_DATA

task = config_file_tree.TASK
sample_target_column = config_file_tree.SAMPLE_TARGET_COL
patient_target_col = config_file_tree.PATIENT_TARGET_COL

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
path_to_clinical = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")

path_train_val_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_train_val.csv")
path_test_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")

path_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR

pretrained_sample_encoder_path = config_file_tree.PRETRAINED_SAMPLE_ENCODER

# OUTPUT DIRECTORIES & PATHS CONFIGURATION
curr_date = datetime.now().strftime('%d_%m_%Y')
output_base_dir = os.path.join("trained_models", curr_date)

# GAT Output Paths
gat_output_dir = os.path.join(output_base_dir, "GAT")
gat_learning_curves_dir = os.path.join(gat_output_dir, "learning_curves")
os.makedirs(gat_learning_curves_dir, exist_ok=True)

gat_global_model_path = os.path.join(gat_output_dir, f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt")
label_encoder_sample_path = os.path.join(gat_output_dir, f"label_encoder_{sample_target_column}.joblib")
metadata_preprocessors_path = os.path.join(gat_output_dir, f"metadata_preprocessors_{sample_target_column}.joblib")
node_scaler_path = os.path.join(gat_output_dir, f"node_scaler_{sample_target_column}.joblib")
gat_report_filename = "report_kfold_test_results_GAT.csv"

# MIL Output Paths
mil_output_dir = os.path.join(output_base_dir, "MIL")
mil_learning_curves_dir = os.path.join(mil_output_dir, "learning_curves")
os.makedirs(mil_learning_curves_dir, exist_ok=True)

mil_global_model_path = os.path.join(mil_output_dir, f"best_MIL_{patient_target_col}_{'withClin' if use_clinical else 'noClin'}.pt")
label_encoder_patient_path = os.path.join(mil_output_dir, f"label_encoder_patient_{patient_target_col}.joblib")
clinical_preprocessors_path = os.path.join(mil_output_dir, f"clinical_preprocessors_{patient_target_col}.joblib")
mil_report_filename = "report_kfold_test_results_MIL.csv"


""""" loading metadata and clinical data """""

df_metadata = pd.read_csv(path_to_metadata, sep="\t")

metadata_train_val_samples = pd.read_csv(path_train_val_csv)
metadata_test_samples = pd.read_csv(path_test_csv)

df_metadata_train_val = df_metadata[
    df_metadata['sample_id'].isin(metadata_train_val_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

df_metadata_test = df_metadata[
    df_metadata['sample_id'].isin(metadata_test_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    cols_to_keep = ['patient_id', patient_target_col]
    if use_clinical:
        cols_to_keep += config_file_tree.CLINICAL_FEATURES  # todo creare clinical_for_train e farlo lì?

    df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
    clinical_patient_id_col = 'patient_id'


""""" target encoding  """""

if task == 'classification':
    label_encoder_sample = sklearn.preprocessing.LabelEncoder()
    label_encoder_sample.fit(df_metadata[sample_target_column].dropna())

    df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(df_metadata_train_val[sample_target_column])
    df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])

    joblib.dump(label_encoder_sample, label_encoder_sample_path)
    logging.info(f"Sample classes identified {sample_target_column}:    " + str(label_encoder_sample.classes_))
else:
    logging.info(f"Starting training on survival task using target event: {sample_target_column}\n")

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).copy()
    label_encoder_patient = sklearn.preprocessing.LabelEncoder()
    df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])

    num_classes_patient_target = len(label_encoder_patient.classes_)
    joblib.dump(label_encoder_patient, label_encoder_patient_path)
    logging.info(f"Patient classes identified for MIL ({patient_target_col}):   {label_encoder_patient.classes_}\n")

    # dict patient_id -> patient_target
    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()


""""" loading graphs """""

'''
logging.info("Train + Val Dataset init...")
train_val_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_train_val, task=task, target_col_metadata=sample_target_column)

logging.info("Test Dataset init...")
test_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_test, task=task, target_col_metadata=sample_target_column)
'''
# IN-MEMORY CACHING (Pre-load in RAM), more efficient
logging.info("Train + Val Dataset init...")
raw_train_val_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_train_val, task=task, target_col=sample_target_column)

logging.info("Test Dataset init...")
raw_test_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_test, task=task, target_col=sample_target_column)

logging.info("Pre-loading datasets into RAM to avoid disk I/O bottleneck...")
train_val_dataset = [raw_train_val_dataset[i] for i in tqdm(range(len(raw_train_val_dataset)), desc="Loading Train/Val")]
test_dataset = [raw_test_dataset[i] for i in tqdm(range(len(raw_test_dataset)), desc="Loading Test")]

print_details(raw_train_val_dataset)


""""" set model_samples_encoder parameters """""

num_genes_features = raw_train_val_dataset.num_features  # multiomics feature vector
logging.info(f"Gene/Node Features found to be used : {num_genes_features}")


# saving un-scaled features to re-scale them to every fold separately
for graph in train_val_dataset:
    graph.x_raw = graph.x.clone()

for graph in test_dataset:
    graph.x_raw = graph.x.clone()


epochs = 100

criterion_samples = None
y_skf = None

if task == 'classification':
    #y_labels_samples = torch.tensor([train_val_dataset[i].y.item() for i in range(len(train_val_dataset))])
    y_labels_samples = torch.tensor(df_metadata_train_val[sample_target_column].values, dtype=torch.long)
    y_skf = y_labels_samples
    unique_labels_samples = torch.unique(y_labels_samples)
    logging.info(f"Unique sample labels found in dataset: {unique_labels_samples}")
    num_classes_samples = len(unique_labels_samples)

    # different weights to classes based on number of samples

    class_counts_samples = torch.bincount(y_labels_samples)
    total_samples = len(y_labels_samples)
    class_weights_samples = total_samples / (len(class_counts_samples) * class_counts_samples.float())
    class_weights_samples = class_weights_samples.to(device)

    logging.info(f"Samples classes counts: {class_counts_samples}")
    logging.info(f"Sample classes weights: {class_weights_samples}\n")

    criterion_samples = torch.nn.CrossEntropyLoss(weight=class_weights_samples)

elif task == 'survival':
    # stratification based on event (0 = censured, 1 =is_platinum_resistant event)
    #y_events = torch.tensor([train_val_dataset[i].y_event.item() for i in range(len(train_val_dataset))])
    y_events = torch.tensor(df_metadata_train_val[sample_target_column].values, dtype=torch.float32)
    y_skf = y_events

    criterion_samples = CoxPHLoss()


""""" k-fold train loop """""

groups = df_metadata_train_val['patient_id'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)

samples_fold_results = []
patients_fold_results = []

global_best_gat_score = float('inf') if task == 'classification' else -float('inf')
global_best_mil_loss = float('inf')

# TODO keep in check
'''# Map patient -> clinical target
patient_to_clinical_target = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

# Composite target for every sample in df_metadata_train_val
composite_y = [
    f"{row[sample_target_column]}_{patient_to_clinical_target[row['patient_id']]}"
    for _, row in df_metadata_train_val.iterrows()
]

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, composite_y, groups=groups)):'''
for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, y_skf, groups=groups)):

    logging.info(f"--- FOLD {fold + 1} ---\n")

    # SAMPLE LEVEL

    logging.info("Computing Node Features scaling parameters using StandardScaler...")

    # train set node concatenation as NumPy array
    train_nodes_np = np.vstack([train_val_dataset[i].x_raw.numpy() for i in train_idx])

    node_scaler = StandardScaler()
    node_scaler.fit(train_nodes_np)

    # scaling and PyTorch Tensor conversion for Train, Val e Test
    for i in train_idx:
        scaled_x = node_scaler.transform(train_val_dataset[i].x_raw.numpy())
        train_val_dataset[i].x = torch.tensor(scaled_x, dtype=torch.float32)

    for i in val_idx:
        scaled_x = node_scaler.transform(train_val_dataset[i].x_raw.numpy())
        train_val_dataset[i].x = torch.tensor(scaled_x, dtype=torch.float32)

    for graph in test_dataset:
        scaled_x = node_scaler.transform(graph.x_raw.numpy())
        graph.x = torch.tensor(scaled_x, dtype=torch.float32)


    """"" metadata fold specific processing """""

    num_sample_features = 0
    fold_metadata_preprocessors = None

    if use_metadata:
        logging.info("Executing fold-specific sample feature encoding and preprocessing...")

        # retrieving samples ids of current train/val splits from graphs sample ids
        df_metadata_train_raw = df_metadata_train_val.iloc[train_idx].copy()
        df_metadata_val_raw = df_metadata_train_val.iloc[val_idx].copy()
        df_metadata_test_raw = df_metadata_test.copy()

        # fit-transform only on train set
        df_train_proc, encoders, scalers, imputers, categories = pre_train_imputation_encoding(
            df=df_metadata_train_raw,
            task=task,
            target_col=sample_target_column,
            is_train=True
        )

        fold_metadata_preprocessors = {
            'encoders': encoders,
            'scalers': scalers,
            'imputers': imputers,
            'categories': categories
        }

        # transform of validation and test
        df_val_proc = pre_train_imputation_encoding(
            df=df_metadata_val_raw,
            task=task,
            target_col=sample_target_column,
            is_train=False,
            encoders=encoders,
            scalers=scalers,
            imputers=imputers,
            categories=categories
        )

        df_test_proc = pre_train_imputation_encoding(
            df=df_metadata_test_raw,
            task=task,
            target_col=sample_target_column,
            is_train=False,
            encoders=encoders,
            scalers=scalers,
            imputers=imputers,
            categories=categories
        )

        # train + val processed e posizionamento di 'sample_id' come Indice
        df_train_val_proc = pd.DataFrame(pd.concat([df_train_proc, df_val_proc], axis=0))
        df_train_val_proc = df_train_val_proc.set_index('sample_id')
        df_test_proc = df_test_proc.set_index('sample_id')

        exclude_cols = {sample_target_column, 'patient_id'}

        sample_feature_cols = [c for c in df_train_val_proc.columns if c not in exclude_cols]
        num_sample_features = len(sample_feature_cols)
        logging.info(f"Processed sample features ({num_sample_features}):   {sample_feature_cols}\n")

        # update metadata_x in graphs loaded in RAM
        update_graphs_metadata(train_val_dataset, df_train_val_proc, sample_feature_cols)
        update_graphs_metadata(test_dataset, df_test_proc, sample_feature_cols)


    """"" model """""

    gat_hidden_channels = 64
    gat_heads = 2

    if task == 'classification':
        model_samples_encoder = GAT(
            num_node_features=num_genes_features,
            num_classes=num_classes_samples,
            hidden_channels=gat_hidden_channels,
            heads=gat_heads,
            num_sample_features=num_sample_features if use_metadata else 0,
        ).to(device)


    """"" fold specific dataset separation and DataLoader """""

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


    if config_file_tree.SAMPLES_LEVEL_TRAIN:
        logging.info(str(model_samples_encoder) + "\n")
        optimizer_samples = torch.optim.AdamW(model_samples_encoder.parameters(), lr=0.001, weight_decay=1e-4)

        if task == 'classification':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_samples, mode='min', factor=0.5, patience=10)
        elif task == 'survival':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_samples, mode='max', factor=0.5, patience=5)


        """"" fold train loop """""

        # metrics to evaluate for improvement
        best_val_loss = float('inf')  # minimizing loss
        best_val_c_index = 0.0  # maximizing C-index
        early_stopping_counter = 0
        history_gat = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        fold_best_gat_weights = None

        for epoch in tqdm(range(1, epochs + 1)):
            if task == 'classification':
                train_loss, train_acc = train_epoch(device, model_samples_encoder, train_loader, optimizer_samples, criterion_samples)
                val_loss, val_metrics = evaluate(device, model_samples_encoder, val_loader, criterion_samples)

                scheduler.step(val_loss)  # update learning rate

                history_gat['train_loss'].append(train_loss)
                history_gat['val_loss'].append(val_loss)
                history_gat['train_metric'].append(train_acc)
                history_gat['val_metric'].append(val_metrics['acc'])

                logging.info(f"[SAMPLES] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
                logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

                if val_loss < best_val_loss:
                    logging.info("IMPROVEMENT!\n")
                    best_val_loss = val_loss
                    fold_best_gat_weights = {k: v.cpu().clone() for k, v in model_samples_encoder.state_dict().items()}
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            elif task == 'survival':
                train_loss = train_epoch_survival(device, model_samples_encoder, train_loader, optimizer_samples, criterion_samples)
                val_loss, val_metrics = evaluate_survival(device, model_samples_encoder, val_loader, criterion_samples)
                val_c_index = val_metrics['c_index']

                scheduler.step(val_c_index)  # update learning rate

                logging.info(
                    f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}"
                )

                if val_c_index > best_val_c_index:
                    logging.info(
                        f"-->IMPROVEMENT! {best_val_c_index:.4f} -> {val_c_index:.4f}\n")
                    best_val_c_index = val_c_index
                    fold_best_gat_weights = {k: v.cpu().clone() for k, v in model_samples_encoder.state_dict().items()}
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            if early_stopping_counter > 20:
                logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
                break

        save_learning_curves(
            history_gat,
            fold_num=fold + 1,
            output_dir=gat_learning_curves_dir
        )

        is_global_best = (best_val_loss < global_best_gat_score) if task == 'classification' else (
                    best_val_c_index > global_best_gat_score)
        if is_global_best and fold_best_gat_weights is not None:
            logging.info(
                f"*** GLOBAL BEST GAT UPDATED! Fold {fold + 1} score ({best_val_loss:.4f}) beat global best ({global_best_gat_score:.4f}). Overwriting {gat_global_model_path} ***\n")
            global_best_gat_score = best_val_loss
            torch.save(fold_best_gat_weights, gat_global_model_path)

            joblib.dump(node_scaler, node_scaler_path)
            logging.info(f"Saved best node scaler to {node_scaler_path}")

            if use_metadata and fold_metadata_preprocessors is not None:
                # saving preprocessors for evaluation/model_samples_encoder usage
                joblib.dump(fold_metadata_preprocessors, metadata_preprocessors_path)
                logging.info(f"Saved best metadata preprocessors to {metadata_preprocessors_path}")

        # Carica i pesi migliori del fold per la valutazione sul Test set del fold corrente
        if fold_best_gat_weights is not None:
            model_samples_encoder.load_state_dict(fold_best_gat_weights)

    else:
        logging.info(
            f"--- Loading Sample Encoder pre-trained model from {pretrained_sample_encoder_path} ---")

        if os.path.exists(pretrained_sample_encoder_path):
            model_to_load = pretrained_sample_encoder_path
        else:
            raise FileNotFoundError(f"Pre-trained model not found at {pretrained_sample_encoder_path}")

        model_samples_encoder.load_state_dict(torch.load(model_to_load, map_location=device))

    # evaluation of loaded model
    model_samples_encoder.eval()
    if task == 'classification':
        test_loss, test_metrics = evaluate(device, model_samples_encoder, test_loader, criterion_samples)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_metrics['acc']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test AUC: {test_metrics['auc']:.4f}\n"
        )
        samples_fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })

    elif task == 'survival':
        test_loss, test_metrics = evaluate_survival(device, model_samples_encoder, test_loader, criterion_samples)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test C-Index: {test_metrics['c_index']:.4f}\n"
        )
        samples_fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'c_index': test_metrics['c_index']
        })


    # PATIENT LEVEL

    if config_file_tree.PATIENTS_LEVEL_TRAIN:
        logging.info('--- PATIENT-LEVEL TRAINING: MIL finetuning on patient target label ---\n')

        patient_clinical_map = {}
        num_patients_features = 0


        """"" patient features fold specific processing """""

        fold_clinical_preprocessors = None
        if use_clinical:
            logging.info("Executing fold-specific patient clinical data preprocessing...")

            # retrieving patients ids of current train/val/test splits
            train_patients_ids = set(df_metadata_train_val.iloc[train_idx]['patient_id'].unique())
            val_patients_ids = set(df_metadata_train_val.iloc[val_idx]['patient_id'].unique())
            test_patients_ids = set(df_metadata_test['patient_id'].unique())

            df_clinical_train_raw = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(train_patients_ids)].copy()
            df_clinical_val_raw = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(val_patients_ids)].copy()
            df_clinical_test_raw = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(test_patients_ids)].copy()

            # clinical features encoding
            df_clinical_train_proc, p_encoders, p_scalers, p_imputers, p_categories = pre_train_imputation_encoding(
                df=df_clinical_train_raw,
                task='classification',
                target_col=patient_target_col,
                is_train=True
            )

            fold_clinical_preprocessors = {
                'encoders': p_encoders,
                'scalers': p_scalers,
                'imputers': p_imputers,
                'categories': p_categories
            }

            df_clinical_val_proc = pre_train_imputation_encoding(
                df=df_clinical_val_raw,
                task='classification',
                target_col=patient_target_col,
                is_train=False,
                encoders=p_encoders,
                scalers=p_scalers,
                imputers=p_imputers,
                categories=p_categories
            )

            df_clinical_test_proc = pre_train_imputation_encoding(
                df=df_clinical_test_raw,
                task='classification',
                target_col=patient_target_col,
                is_train=False,
                encoders=p_encoders,
                scalers=p_scalers,
                imputers=p_imputers,
                categories=p_categories
            )

            df_clinical_all_proc = pd.DataFrame(
                pd.concat([df_clinical_train_proc, df_clinical_val_proc, df_clinical_test_proc], axis=0))
            df_clinical_all_proc = df_clinical_all_proc.set_index(clinical_patient_id_col)

            exclude_patients_cols = {patient_target_col, 'patient_id'}
            patients_feature_cols = [c for c in df_clinical_all_proc.columns if c not in exclude_patients_cols]
            num_patients_features = len(patients_feature_cols)
            logging.info(f"Number of processed patient clinical features for Fold {fold + 1}: {num_patients_features}")

            # creation dict {patient_id: tensor_features}
            for patient_id in df_clinical_all_proc.index:
                feats = df_clinical_all_proc.loc[patient_id, patients_feature_cols].values.astype(np.float32)
                patient_clinical_map[patient_id] = torch.tensor(feats, dtype=torch.float32)


        """"" fold specific dataset separation and DataLoader """""

        train_samples = [train_val_dataset[i] for i in train_idx]
        val_samples = [train_val_dataset[i] for i in val_idx]

        # Patient Bag Dataset creation
        patient_train_dataset = PatientBagDataset(
            train_samples,
            patient_target_map,
            patient_clinical_map=patient_clinical_map if use_clinical else None
        )
        patient_val_dataset = PatientBagDataset(
            val_samples,
            patient_target_map,
            patient_clinical_map=patient_clinical_map if use_clinical else None
        )
        patient_test_dataset = PatientBagDataset(
            test_dataset,
            patient_target_map,
            patient_clinical_map=patient_clinical_map if use_clinical else None
        )

        # DataLoader MIL: 1 patient per batch
        patients_train_loader = DataLoader(patient_train_dataset, batch_size=1, shuffle=True)
        patients_val_loader = DataLoader(patient_val_dataset, batch_size=1, shuffle=False)
        patients_test_loader = DataLoader(patient_test_dataset, batch_size=1, shuffle=False)


        """"" model """""

        train_patients_targets = torch.tensor([patient_target_map[patient_id] for patient_id in patient_train_dataset.patient_ids],
                                              dtype=torch.long)

        class_counts_patients = torch.bincount(train_patients_targets, minlength=num_classes_patient_target)
        class_counts_patients = torch.clamp(class_counts_patients, min=1)  # TODO da lasciare? per prevenire divisioni a 0

        class_weights_patients = (len(train_patients_targets) / (len(class_counts_patients) * class_counts_patients.float())).to(device)
        criterion_patients = torch.nn.CrossEntropyLoss(weight=class_weights_patients)

        logging.info(f"Patient classes counts: {class_counts_patients}")
        logging.info(f"Patient classes weights: {class_weights_patients}\n")

        model_patient_encoder = PatientMILModel(
            sample_encoder=model_samples_encoder,  # passing samples pre-trained encoder
            num_classes=num_classes_patient_target,
            num_clinical_features=num_patients_features if use_clinical else 0
        ).to(device)
        logging.info(str(model_patient_encoder) + "\n")

        # different Learning Rate, lower for GAT (to keep learned features) and higher for Pooling and Classifier
        optimizer_patients = torch.optim.AdamW(
            [
                {'params': model_patient_encoder.sample_encoder.parameters(), 'lr': 1e-5},
                {'params': model_patient_encoder.pooling.parameters(), 'lr': 1e-3},
                {'params': model_patient_encoder.classifier.parameters(), 'lr': 1e-3},
            ],
            weight_decay=1e-4,
        )

        # Epoch Loop MIL Fine-Tuning
        best_val_loss = float('inf')
        early_stopping_counter = 0
        history_mil = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        fold_best_mil_weights = None

        for epoch in tqdm(range(1, epochs + 1)):
            train_loss, train_acc = train_epoch_mil(device, model_patient_encoder, patients_train_loader, optimizer_patients, criterion_patients)
            val_loss, val_metrics = evaluate_mil(device, model_patient_encoder, patients_val_loader, criterion_patients)

            history_mil['train_loss'].append(train_loss)
            history_mil['val_loss'].append(val_loss)
            history_mil['train_metric'].append(train_acc)
            history_mil['val_metric'].append(val_metrics['acc'])

            logging.info(f"[PATIENTS] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
            logging.info(f"Val metrics: Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}"
            )

            if val_loss < best_val_loss:
                logging.info("PATIENT MODEL IMPROVEMENT!\n")
                best_val_loss = val_loss
                fold_best_mil_weights = {k: v.cpu().clone() for k, v in model_patient_encoder.state_dict().items()}
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter > 15:
                logging.info(f"Patients model Early Stopping triggered at epoch {epoch}\n")
                break

        save_learning_curves(
            history_mil,
            fold_num=fold + 1,
            output_dir=mil_learning_curves_dir
        )

        # Test Evaluation on best patient encoder model
        if best_val_loss < global_best_mil_loss and fold_best_mil_weights is not None:
            logging.info(
                f"*** GLOBAL BEST MIL UPDATED! Fold {fold + 1} loss ({best_val_loss:.4f}) beat global best ({global_best_mil_loss:.4f}). Overwriting {mil_global_model_path} ***\n")
            global_best_mil_loss = best_val_loss
            torch.save(fold_best_mil_weights, mil_global_model_path)

            if use_clinical and fold_clinical_preprocessors is not None:
                # saving preprocessors for evaluation/model_patients usage
                joblib.dump(fold_clinical_preprocessors, clinical_preprocessors_path)
                logging.info(f"Saved best clinical preprocessors to {clinical_preprocessors_path}")
        if fold_best_mil_weights is not None:
            model_patient_encoder.load_state_dict(fold_best_mil_weights)

        test_loss, test_metrics = evaluate_mil(device, model_patient_encoder, patients_test_loader, criterion_patients)

        logging.info(
            f"[PATIENTS-PHASE] Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Acc: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}\n"
        )
        patients_fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })

""""" folds evaluation summary """""

logging.info("--- SAMPLE LEVEL RESULTS ---")
show_fold_results(pd.DataFrame(samples_fold_results), task, save_to_csv=True, dir_to_save=gat_output_dir, filename=gat_report_filename)

if config_file_tree.PATIENTS_LEVEL_TRAIN and patients_fold_results:
    logging.info("--- PATIENT LEVEL RESULTS ---")
    show_fold_results(pd.DataFrame(patients_fold_results), task, save_to_csv=True, dir_to_save=mil_output_dir, filename=mil_report_filename)

logging.info("DONE\n")