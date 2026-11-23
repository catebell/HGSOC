import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from PatientBagDataset import PatientBagDataset
from models.GAT import GAT
from models.PatientMIL_regularized import PatientMILModel
from utils_train_eval import (
    print_details,
    pre_train_imputation_encoding,
    save_training_config_report,
    handle_pretrained_gat_checkpoint
)
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from train_functions import train_epoch, evaluate, show_fold_results, train_epoch_mil, evaluate_mil, \
    save_learning_curves, \
    save_patient_classification_results, SelectiveNodeScaler

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

gat_pretrained_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
pretrained_sample_encoder_path = os.path.join(
    gat_pretrained_dir,
    f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt"
)

# OUTPUT DIRECTORIES & PATHS CONFIGURATION
curr_date = datetime.now().strftime('hybrid_MIL_platRes_PDS_fixed_for_cox_%d_%m_%Y')
output_base_dir = os.path.join("trained_models", curr_date)
os.makedirs(output_base_dir, exist_ok=True)

_, config_text = save_training_config_report(output_base_dir)

# GAT Output Paths
gat_output_dir = os.path.join(output_base_dir, "GAT")
gat_learning_curves_dir = os.path.join(gat_output_dir, "learning_curves")
# if config_file_tree.SAMPLES_LEVEL_TRAIN:
os.makedirs(gat_learning_curves_dir, exist_ok=True)

gat_global_model_path = os.path.join(gat_output_dir,
                                     f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt")
label_encoder_sample_path = os.path.join(gat_output_dir, f"label_encoder_{sample_target_column}.joblib")
metadata_preprocessors_path = os.path.join(gat_output_dir, f"metadata_preprocessors_{sample_target_column}.joblib")
node_scaler_path = os.path.join(gat_output_dir, f"node_scaler_{sample_target_column}.joblib")
gat_report_filename = "report_kfold_test_results_GAT.csv"

# MIL Output Paths
mil_output_dir = os.path.join(output_base_dir, "MIL")
mil_learning_curves_dir = os.path.join(mil_output_dir, "learning_curves")
if config_file_tree.PATIENTS_LEVEL_TRAIN:
    os.makedirs(mil_learning_curves_dir, exist_ok=True)

    if not config_file_tree.SAMPLES_LEVEL_TRAIN:
        handle_pretrained_gat_checkpoint(
            gat_pretrained_dir=gat_pretrained_dir,
            gat_output_dir=gat_output_dir,
            config_text=config_text
        )

mil_global_model_path = os.path.join(mil_output_dir,
                                     f"best_MIL_{patient_target_col}_{'withClin' if use_clinical else 'noClin'}.pt")
label_encoder_patient_path = os.path.join(mil_output_dir, f"label_encoder_patient_{patient_target_col}.joblib")
clinical_preprocessors_path = os.path.join(mil_output_dir, f"clinical_preprocessors_{patient_target_col}.joblib")
mil_report_filename = "report_kfold_test_results_MIL.csv"

active_feature_names = [config_file_tree.NODE_FEATURE_NAMES[i] for i in config_file_tree.NODE_FEATURE_INDEX]

""""" logging configuration """""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_base_dir, 'last_execution.log'), mode='w'),
        logging.StreamHandler()
    ],
    force=True
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
        cols_to_keep += config_file_tree.CLINICAL_FEATURES

    df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
    clinical_patient_id_col = 'patient_id'

""""" target encoding  """""

if config_file_tree.SAMPLES_LEVEL_TRAIN:
    label_encoder_sample = sklearn.preprocessing.LabelEncoder()
    label_encoder_sample.fit(df_metadata[sample_target_column].dropna())

    df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(
        df_metadata_train_val[sample_target_column])
    df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])

    joblib.dump(label_encoder_sample, label_encoder_sample_path)
    logging.info(f"Sample classes identified {sample_target_column}:    " + str(label_encoder_sample.classes_))
else:
    if os.path.exists(label_encoder_sample_path):
        label_encoder_sample = joblib.load(label_encoder_sample_path)
        df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(
            df_metadata_train_val[sample_target_column])
        df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])
        sample_class_names = [str(c) for c in label_encoder_sample.classes_]
    else:
        logging.warning(
            f"[WARNING] Sample label encoder joblib not found in: '{label_encoder_sample_path}'. New LabelEncoder() fallback.")
        label_encoder_sample = LabelEncoder()
        label_encoder_sample.fit(df_metadata[sample_target_column].dropna())

        df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(
            df_metadata_train_val[sample_target_column])
        df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])

        joblib.dump(label_encoder_sample, label_encoder_sample_path)
        logging.info(f"Sample classes identified {sample_target_column}:    " + str(label_encoder_sample.classes_))

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).copy()
    label_encoder_patient = sklearn.preprocessing.LabelEncoder()
    df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])

    num_classes_patient_target = len(label_encoder_patient.classes_)
    joblib.dump(label_encoder_patient, label_encoder_patient_path)
    logging.info(f"Patient classes identified for MIL ({patient_target_col}):   {label_encoder_patient.classes_}\n")

    # =========================================================================
    if config_file_tree.RUN_PERMUTATION_TEST:
        seed_perm = 42
        logging.info("=======================================================")
        logging.info(f"[PERMUTATION TEST] ACTIVE (seed={seed_perm})")

        # Rimescolamento casuale delle etichette dei pazienti
        np.random.seed(seed_perm)
        original_labels = df_clinical_clean[patient_target_col].values.copy()
        df_clinical_clean[patient_target_col] = np.random.permutation(original_labels)

        logging.info(f"Distribuzione etichette prima: {np.bincount(original_labels)}")
        logging.info(f"Distribuzione etichette permutate: {np.bincount(df_clinical_clean[patient_target_col].values)}")
        logging.info("=======================================================\n")
    # =========================================================================

    # dict patient_id -> patient_target
    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

""""" loading graphs """""

# IN-MEMORY CACHING (Pre-load in RAM), more efficient
logging.info("Train + Val Dataset init...")
raw_train_val_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_train_val, task=task,
                                             target_col=sample_target_column)

logging.info("Test Dataset init...")
raw_test_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_test, task=task, target_col=sample_target_column)

logging.info("Pre-loading datasets into RAM to avoid disk I/O bottleneck...\n")
train_val_dataset = [raw_train_val_dataset[i] for i in
                     tqdm(range(len(raw_train_val_dataset)), desc="Loading Train/Val")]
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

epochs = 50
epochs_patients = 30

criterion_samples = None
y_skf = None

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

""""" stratified k-fold train loop """""

groups = df_metadata_train_val['patient_id'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)

samples_fold_results = []
patients_fold_results = []

global_best_gat_score = float('inf') if task == 'classification' else -float('inf')
global_best_mil_score = float('inf')

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    # combine targets per stratified splits
    # Map patient -> clinical target
    patient_to_clinical_target = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

    # Composite target for every sample in df_metadata_train_val
    composite_y = [
        f"{row[sample_target_column]}_{patient_to_clinical_target.get(row['patient_id'], 'unfiltered')}"
        for _, row in df_metadata_train_val.iterrows()
    ]
    y_skf = composite_y

samples_fold_results = []
patients_fold_results = []
subgroup_fold_results = []  # Added for tracking NACT vs PDS across folds
mil_oof_records = []
for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, y_skf, groups=groups)):
    logging.info(f"--- FOLD {fold + 1} ---\n")

    # SAMPLE LEVEL

    logging.info("Computing Node Features scaling parameters using SelectiveNodeScaler...")

    # train set node concatenation as NumPy array
    train_nodes_np = np.vstack([train_val_dataset[i].x_raw.numpy() for i in train_idx])
    node_scaler = SelectiveNodeScaler(active_feature_names=active_feature_names)
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
            df=df_metadata_train_raw, task=task, target_col=sample_target_column, is_train=True
        )

        fold_metadata_preprocessors = {
            'encoders': encoders, 'scalers': scalers,
            'imputers': imputers, 'categories': categories
        }

        # transform of validation and test
        df_val_proc = pre_train_imputation_encoding(
            df=df_metadata_val_raw, task=task, target_col=sample_target_column, is_train=False,
            encoders=encoders, scalers=scalers, imputers=imputers, categories=categories
        )

        df_test_proc = pre_train_imputation_encoding(
            df=df_metadata_test_raw, task=task, target_col=sample_target_column, is_train=False,
            encoders=encoders, scalers=scalers, imputers=imputers, categories=categories
        )

        # train + val processed e posizionamento di 'sample_id' come Indice
        df_train_val_proc = pd.DataFrame(pd.concat([df_train_proc, df_val_proc], axis=0)).set_index('sample_id')
        df_test_proc = df_test_proc.set_index('sample_id')

        exclude_cols = {sample_target_column, 'patient_id'}

        sample_feature_cols = [c for c in df_train_val_proc.columns if c not in exclude_cols]
        num_sample_features = len(sample_feature_cols)
        logging.info(f"Processed sample features ({num_sample_features}):   {sample_feature_cols}\n")

        # update metadata_x in graphs loaded in RAM
        update_graphs_metadata(train_val_dataset, df_train_val_proc, sample_feature_cols)
        update_graphs_metadata(test_dataset, df_test_proc, sample_feature_cols)

    else:
        logging.info("Metadata extra data exluded.")

    """"" fold specific model initialization + dataset separation"""""

    num_nodes_per_graph = raw_train_val_dataset[0].x.shape[0]
    model_samples_encoder = GAT(
        num_node_features=num_genes_features,
        num_classes=num_classes_samples,
        num_nodes=num_nodes_per_graph,
        num_sample_features=num_sample_features if use_metadata else 0,
    ).to(device)

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # SAMPLES ENCODER TRAINING

    if config_file_tree.SAMPLES_LEVEL_TRAIN:
        logging.info(str(model_samples_encoder) + "\n")
        optimizer_samples = torch.optim.AdamW(model_samples_encoder.parameters(), lr=0.001, weight_decay=1e-3)

        scheduler_samples = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_samples, mode='min', factor=0.5,
                                                                       patience=10)

        """"" fold train loop """""

        # metrics to evaluate for improvement
        best_val_loss = float('inf')  # minimizing loss
        best_val_c_index = 0.0  # maximizing C-index
        early_stopping_counter = 0
        history_gat = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        fold_best_gat_weights = None

        for epoch in tqdm(range(1, epochs + 1), desc=f"GAT Fold {fold + 1}"):
            train_loss, train_acc = train_epoch(device, model_samples_encoder, train_loader, optimizer_samples,
                                                criterion_samples)
            val_loss, val_metrics = evaluate(device, model_samples_encoder, val_loader, criterion_samples)

            scheduler_samples.step(val_loss)  # update learning rate

            history_gat['train_loss'].append(train_loss)
            history_gat['val_loss'].append(val_loss)
            history_gat['train_metric'].append(train_acc)
            history_gat['val_metric'].append(val_metrics['acc'])

            logging.info(
                f"[SAMPLES] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
            logging.info(
                f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

            if val_loss < best_val_loss:
                logging.info("IMPROVEMENT!\n")
                best_val_loss = val_loss
                fold_best_gat_weights = {k: v.cpu().clone() for k, v in model_samples_encoder.state_dict().items()}
                early_stopping_counter = 0

                if best_val_loss < global_best_gat_score:
                    logging.info(
                        f"*** GLOBAL BEST GAT UPDATED! Fold {fold + 1} score ({best_val_loss:.4f}) beat global best ({global_best_gat_score:.4f}). Overwriting {gat_global_model_path} ***\n")
                    global_best_gat_score = best_val_loss
                    torch.save(fold_best_gat_weights, gat_global_model_path)

                    joblib.dump(node_scaler, node_scaler_path)
                    logging.info(f"Saved best node scaler to {node_scaler_path}\n")

                    if use_metadata and fold_metadata_preprocessors is not None:
                        # saving preprocessors for evaluation/model_samples_encoder usage
                        joblib.dump(fold_metadata_preprocessors, metadata_preprocessors_path)
                        logging.info(f"Saved best metadata preprocessors to {metadata_preprocessors_path}\n")
            else:
                early_stopping_counter += 1

            if early_stopping_counter > 15:
                logging.info("--- [Early stopping triggered] ---\n")
                break

        save_learning_curves(history_gat, fold_num=fold + 1, output_dir=gat_learning_curves_dir)

        # load best weights of current fold for test set validation
        if fold_best_gat_weights is not None:
            model_samples_encoder.load_state_dict(fold_best_gat_weights)

        # evaluation of sample encoder in fold
        model_samples_encoder.eval()

        # Conteggio dei campioni nel test set per ogni classe
        test_sample_targets = [g.y.item() for g in test_dataset]
        sample_counts = np.bincount(test_sample_targets, minlength=len(label_encoder_sample.classes_))

        fold_sample_dict = {
            'fold': fold + 1,
            'total_samples': len(test_sample_targets)
        }
        for cls_idx, cls_name in enumerate(label_encoder_sample.classes_):
            fold_sample_dict[f'n_{cls_name}'] = int(sample_counts[cls_idx])

        test_loss, test_metrics = evaluate(device, model_samples_encoder, test_loader, criterion_samples)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_metrics['acc']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test AUC: {test_metrics['auc']:.4f}\n"
        )
        fold_sample_dict.update({
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })
        samples_fold_results.append(fold_sample_dict)
    else:
        logging.info(
            f"##### Loading pre-trained Sample Encoder checkpoint from {pretrained_sample_encoder_path} #####\n")

        if os.path.exists(pretrained_sample_encoder_path):
            model_samples_encoder.load_state_dict(torch.load(pretrained_sample_encoder_path, map_location=device))
        else:
            raise FileNotFoundError(f"Pre-trained model not found at {pretrained_sample_encoder_path}")

    # PATIENT LEVEL

    if config_file_tree.PATIENTS_LEVEL_TRAIN:
        logging.info('--- PATIENT-LEVEL TRAINING: MIL fine-tuning on patient target label ---\n')

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

            df_clinical_train_raw = df_clinical_clean[
                df_clinical_clean[clinical_patient_id_col].isin(train_patients_ids)].copy()
            df_clinical_val_raw = df_clinical_clean[
                df_clinical_clean[clinical_patient_id_col].isin(val_patients_ids)].copy()
            df_clinical_test_raw = df_clinical_clean[
                df_clinical_clean[clinical_patient_id_col].isin(test_patients_ids)].copy()

            # clinical features encoding
            df_clinical_train_proc, p_encoders, p_scalers, p_imputers, p_categories = pre_train_imputation_encoding(
                df=df_clinical_train_raw, task='classification', target_col=patient_target_col, is_train=True
            )

            fold_clinical_preprocessors = {
                'encoders': p_encoders, 'scalers': p_scalers,
                'imputers': p_imputers, 'categories': p_categories
            }

            df_clinical_val_proc = pre_train_imputation_encoding(
                df=df_clinical_val_raw, task='classification', target_col=patient_target_col, is_train=False,
                encoders=p_encoders, scalers=p_scalers, imputers=p_imputers, categories=p_categories
            )

            df_clinical_test_proc = pre_train_imputation_encoding(
                df=df_clinical_test_raw, task='classification', target_col=patient_target_col, is_train=False,
                encoders=p_encoders, scalers=p_scalers, imputers=p_imputers, categories=p_categories
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

        else:
            logging.info("Clinical extra data excluded.")

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

        patients_train_loader = DataLoader(patient_train_dataset, batch_size=8, shuffle=True)
        patients_val_loader = DataLoader(patient_val_dataset, batch_size=8, shuffle=False)
        patients_test_loader = DataLoader(patient_test_dataset, batch_size=8, shuffle=False)

        """"" dataset separation fold specific model """""

        train_patients_targets = torch.tensor(
            [patient_target_map[patient_id] for patient_id in patient_train_dataset.patient_ids],
            dtype=torch.long)

        class_counts_patients = torch.bincount(train_patients_targets, minlength=num_classes_patient_target)
        class_counts_patients = torch.clamp(class_counts_patients, min=1)

        class_weights_patients = (
                    len(train_patients_targets) / (len(class_counts_patients) * class_counts_patients.float())).to(
            device)
        criterion_patients = torch.nn.CrossEntropyLoss(weight=class_weights_patients)

        logging.info(f"Patient classes counts: {class_counts_patients}")
        logging.info(f"Patient classes weights: {class_weights_patients}\n")

        model_patient_encoder = PatientMILModel(
            sample_encoder=model_samples_encoder,
            num_classes=num_classes_patient_target,
            num_tissue_classes=num_classes_samples,
            proj_dim=24,
            pooling_hidden_dim=16,
            classifier_hidden_dim=16,
            num_clinical_features=num_patients_features if use_clinical else 0,
            dropout=0.3,
            subsample_rate=1.0
        ).to(device)

        logging.info(str(model_patient_encoder) + "\n")

        # 1. FASE DI WARMUP: GAT completamente congelata all'inizio
        warmup_epochs = 8
        # top_gat_layer_names = ('conv2', 'norm2', 'readout')
        top_gat_layer_names = ('norm2', 'readout')
        model_patient_encoder.freeze_sample_encoder()

        # Separa i parametri: testa MIL vs soli strati terminali della GAT
        sample_encoder_params = set(model_patient_encoder.sample_encoder.parameters())
        mil_head_params = [p for p in model_patient_encoder.parameters() if p not in sample_encoder_params]

        top_gat_params = []
        for layer_name in top_gat_layer_names:
            layer = getattr(model_patient_encoder.sample_encoder, layer_name, None)
            if layer is not None:
                top_gat_params.extend(list(layer.parameters()))

        # Nota: durante il warmup top_gat_params hanno requires_grad=False (grad=None),
        # quindi AdamW li ignora senza applicare weight_decay finché non vengono sbloccati.
        optimizer_patients = torch.optim.AdamW(
            [
                {'params': mil_head_params, 'lr': 1e-3, 'weight_decay': 2e-2},
                {'params': top_gat_params, 'lr': 5e-6, 'weight_decay': 1e-2},
            ]
        )

        scheduler_patients = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_patients, T_max=epochs_patients, eta_min=1e-6
        )

        """"" MIL Fine-Tuning Epoch Loop """""

        best_selection_score = float('inf')
        ema_val_loss = None
        ema_alpha = 0.6  # smussa i picchi rumorosi della Val Loss su N=21 pazienti
        min_save_epoch = 3  # evita di salvare modelli immaturi all'Epoca 1-2
        early_stopping_counter = 0

        history_mil = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        fold_best_mil_weights = None

        for epoch in tqdm(range(1, epochs_patients + 1), desc=f"MIL Fold {fold + 1}"):
            # 2. UNFREEZE PARZIALE: dopo il warmup sblocca solo conv2, norm2 e readout
            if epoch == warmup_epochs + 1:
                logging.info(f"[Epoch {epoch:02d}] Warmup completato: unfreeze selettivo di {top_gat_layer_names}\n")
                model_patient_encoder.unfreeze_top_gat_layers(top_gat_layer_names)
                early_stopping_counter = 0  # concede piena pazienza alla fase di fine-tuning GAT

            train_loss, train_acc = train_epoch_mil(
                device, model_patient_encoder, patients_train_loader, optimizer_patients, criterion_patients
            )
            val_loss, val_metrics, _ = evaluate_mil(
                device, model_patient_encoder, patients_val_loader, criterion_patients
            )

            scheduler_patients.step()

            # Calcolo EMA Val Loss e score composito (Loss smussata + ranking AUC)
            ema_val_loss = val_loss if ema_val_loss is None else (
                    ema_alpha * val_loss + (1.0 - ema_alpha) * ema_val_loss
            )
            selection_score = ema_val_loss - 0.15 * val_metrics['auc']

            history_mil['train_loss'].append(train_loss)
            history_mil['val_loss'].append(val_loss)
            history_mil['train_metric'].append(train_acc)
            history_mil['val_metric'].append(val_metrics['acc'])

            logging.info(
                f"[PATIENTS] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} (EMA: {ema_val_loss:.4f})"
            )
            logging.info(
                f"Val metrics: Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f} | Score: {selection_score:.4f}"
            )

            if epoch >= min_save_epoch and selection_score < best_selection_score:
                logging.info("PATIENT MODEL IMPROVEMENT!\n")
                best_selection_score = selection_score
                fold_best_mil_weights = {k: v.cpu().clone() for k, v in model_patient_encoder.state_dict().items()}
                early_stopping_counter = 0

                if best_selection_score < global_best_mil_score and fold_best_mil_weights is not None:
                    logging.info(
                        f"*** GLOBAL BEST MIL UPDATED! Fold {fold + 1} score ({best_selection_score:.4f}) "
                        f"beat global best ({global_best_mil_score:.4f}). Overwriting {mil_global_model_path} ***\n"
                    )
                    global_best_mil_score = best_selection_score
                    torch.save(fold_best_mil_weights, mil_global_model_path)

                    if use_clinical and fold_clinical_preprocessors is not None:
                        joblib.dump(fold_clinical_preprocessors, clinical_preprocessors_path)
                        logging.info(f"Saved best clinical preprocessors to {clinical_preprocessors_path}\n")
            elif epoch > warmup_epochs:
                # L'early stopping parte solo dopo la fase di warmup
                early_stopping_counter += 1

            if early_stopping_counter > 10:
                logging.info(f"Patients model Early Stopping triggered at epoch {epoch}\n")
                break

        save_learning_curves(history_mil, fold_num=fold + 1, output_dir=mil_learning_curves_dir)

        if fold_best_mil_weights is not None:
            model_patient_encoder.load_state_dict(fold_best_mil_weights)  # loading best fold weights for testing

        test_loss, test_metrics, df_attention_fold = evaluate_mil(device, model_patient_encoder, patients_test_loader,
                                                                  criterion_patients)

        attn_save_path = os.path.join(mil_output_dir, f"attention_weights_fold_{fold + 1}.csv")
        df_attention_fold.to_csv(attn_save_path, index=False)

        # patient counts in test set for each fold class
        test_patient_targets = [patient_target_map[pid] for pid in patient_test_dataset.patient_ids]
        patient_counts = np.bincount(test_patient_targets, minlength=len(label_encoder_patient.classes_))

        fold_patient_dict = {
            'fold': fold + 1,
            'total_patients': len(test_patient_targets)
        }
        for cls_idx, cls_name in enumerate(label_encoder_patient.classes_):
            fold_patient_dict[f'n_{cls_name}'] = int(patient_counts[cls_idx])

        fold_patient_dict.update({
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })

        logging.info(
            f"[PATIENTS-PHASE] Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Acc: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}\n"
        )
        patients_fold_results.append(fold_patient_dict)

        _, _, _, val_preds_bundle = evaluate_mil(
            device, model_patient_encoder, patients_val_loader, criterion_patients, return_predictions=True
        )
        val_y_true = val_preds_bundle["y_true"]
        val_y_pred = val_preds_bundle["y_pred"]
        val_y_prob = val_preds_bundle["y_prob"][:, 1] if val_preds_bundle["y_prob"].shape[1] > 1 else \
        val_preds_bundle["y_prob"][:, 0]

        for pid, yt, yp, prob1 in zip(patient_val_dataset.patient_ids, val_y_true, val_y_pred, val_y_prob):
            mil_oof_records.append({
                "fold": fold + 1,
                "patient_id": str(pid).strip(),
                "target_label": int(yt),
                "target_label_name": str(label_encoder_patient.classes_[int(yt)]),
                "mil_prob_resistant": float(prob1),
                "mil_pred": int(yp),
            })

        # saving predictions with patient clinical features and subgroups analysis
        _, _, subgroup_metrics = save_patient_classification_results(
            device=device,
            model=model_patient_encoder,
            test_loader=patients_test_loader,
            label_encoder_patient=label_encoder_patient,
            label_encoder_sample=label_encoder_sample,
            fold_num=fold + 1,
            output_dir=mil_output_dir,
            clinical_features_to_include=['Treatment_strategy', 'PFI_KaplanM_allHGSC', 'Residual_tumor', 'is_hrd']
        )

        for group_name, g_metrics in subgroup_metrics.items():
            subgroup_fold_results.append({
                'fold': fold + 1,
                'subgroup': group_name,
                'n_patients': g_metrics['n_patients'],
                'acc': g_metrics['acc'],
                'f1': g_metrics['f1'],
                'auc': g_metrics['auc']
            })
        # -------------------------------------------------------------------

""""" folds evaluation summary """""

if config_file_tree.SAMPLES_LEVEL_TRAIN and samples_fold_results:
    logging.info("--- SAMPLE LEVEL RESULTS ---")
    show_fold_results(pd.DataFrame(samples_fold_results), task, save_to_csv=True, dir_to_save=gat_output_dir,
                      filename=gat_report_filename)

if config_file_tree.PATIENTS_LEVEL_TRAIN and patients_fold_results:
    logging.info("--- PATIENT LEVEL RESULTS ---")
    show_fold_results(pd.DataFrame(patients_fold_results), task, save_to_csv=True, dir_to_save=mil_output_dir,
                      filename=mil_report_filename)

if config_file_tree.PATIENTS_LEVEL_TRAIN and subgroup_fold_results:
    logging.info("--- SUBGROUP LEVEL RESULTS (NACT vs PDS Across Folds) ---")
    df_subgroups = pd.DataFrame(subgroup_fold_results)

    subgroup_report_path = os.path.join(mil_output_dir, "report_kfold_subgroups_MIL.csv")
    df_subgroups.to_csv(subgroup_report_path, index=False)

    # Compute mean and variance per subgroup
    for grp, df_grp in df_subgroups.groupby('subgroup'):
        mean_auc = df_grp['auc'].mean()
        var_auc = df_grp['auc'].var()
        mean_f1 = df_grp['f1'].mean()
        var_f1 = df_grp['f1'].var()
        logging.info(
            f"Subgroup [{grp}] Overall Performance across folds | "
            f"AUC: {mean_auc:.4f} ± {var_auc:.4f} | F1: {mean_f1:.4f} ± {var_f1:.4f}"
        )

if config_file_tree.PATIENTS_LEVEL_TRAIN and mil_oof_records:
    df_mil_oof = pd.DataFrame(mil_oof_records)
    full_clin_path = os.path.join(config_file_tree.PREPROCESSED_DATA_DIR, "preprocessed_clinical_data.tsv")
    if os.path.exists(full_clin_path):
        df_clin_annot = pd.read_csv(full_clin_path, sep="\t")
        df_clin_annot["patient_id"] = df_clin_annot["patient_id"].astype(str).str.strip()
        wanted = ["patient_id", "Treatment_strategy", "PFI_KaplanM_allHGSC", "Residual_tumor", "is_hrd"]
        df_clin_annot = df_clin_annot[[c for c in wanted if c in df_clin_annot.columns]].drop_duplicates("patient_id")
        df_mil_oof = df_mil_oof.merge(df_clin_annot, on="patient_id", how="left")

    oof_mil_path = os.path.join(mil_output_dir, "oof_patient_predictions.tsv")
    df_mil_oof.to_csv(oof_mil_path, sep="\t", index=False)
    logging.info(f"Saved Neural MIL OOF predictions to: {oof_mil_path}")

logging.info("DONE\n")