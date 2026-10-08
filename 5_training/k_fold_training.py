import logging
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from PatientBagDataset import PatientBagDataset
from models.GAT import GAT
from models.PatientMIL import PatientMILModel
from train_eval_utils import print_details, pre_train_imputation_encoding
from PatientSampleDataset import PatientSampleDataset
from models.MultiOmicGATSurvival import MultiOmicGATSurvival, CoxPHLoss
from train_functions import train_epoch, evaluate, train_epoch_survival, evaluate_survival, show_fold_results, \
    train_epoch_mil, evaluate_mil

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

logging.info("Device: " + str(device))


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

output_dir = os.path.join("trained_models")
os.makedirs(output_dir, exist_ok=True)


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
    cols_to_keep = ['Patient_cohort_code_Patient_Card', patient_target_col]
    if use_clinical:
        cols_to_keep += config_file_tree.CLINICAL_FEATURES  # todo creare clinical_for_train e farlo lì?

    df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
    clinical_patient_id_col = df_clinical.columns[0]


""""" target encoding  """""

if task == 'classification':
    logging.info(f"Starting training for classification task using target feature: {sample_target_column}\n")

    label_encoder_sample = sklearn.preprocessing.LabelEncoder()
    label_encoder_sample.fit(df_metadata[sample_target_column].dropna())

    df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(df_metadata_train_val[sample_target_column])
    df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])

    #joblib.dump(label_encoder_sample, os.path.join(output_dir, f"label_encoder_{sample_target_column}.joblib"))
    logging.info("Classes identified:" + str(label_encoder_sample.classes_))
else:
    logging.info(f"Starting training on survival task using target event: {sample_target_column}\n")

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).copy()
    label_encoder_patient = sklearn.preprocessing.LabelEncoder()
    df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])

    num_classes_patient_target = len(label_encoder_patient.classes_)
    #joblib.dump(label_encoder_patient, os.path.join(output_dir, f"label_encoder_patient_{patient_target_col}.joblib"))
    logging.info(f"Patient classes identified for MIL ({patient_target_col}): {label_encoder_patient.classes_}")

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
logging.info(f"Gene/Node Features found : {num_genes_features}")

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

    logging.info("Samples classes counts: " + str(class_counts_samples))
    logging.info("Sample classes weights: " + str(class_weights_samples))

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

# TODO keep in check
# Map patient -> clinical target
patient_to_clinical_target = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

# Composite target for every sample in df_metadata_train_val
composite_y = [
    f"{row[sample_target_column]}_{patient_to_clinical_target[row['patient_id']]}"
    for _, row in df_metadata_train_val.iterrows()
]

#for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, y_skf, groups=groups)):
for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, composite_y, groups=groups)):
    logging.info(f"--- FOLD {fold + 1} ---\n")

    # SAMPLE LEVEL

    """"" metadata fold specific processing """""

    num_sample_features = 0

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

        # saving preprocessors for evaluation/model_samples_encoder usage
        save_path = os.path.join(output_dir, f"metadata_preprocessors_{task}_fold{fold + 1}.joblib")
        joblib.dump({
            'encoders': encoders,
            'scalers': scalers,
            'imputers': imputers,
            'categories': categories
        }, save_path)

        # train + val processed e posizionamento di 'sample_id' come Indice
        df_train_val_proc = pd.DataFrame(pd.concat([df_train_proc, df_val_proc], axis=0))

        df_train_val_proc = df_train_val_proc.set_index('sample_id')
        df_test_proc = df_test_proc.set_index('sample_id')

        # 4. Selezione delle sole colonne di feature numeriche elaborate TODO necessario?
        exclude_cols = {sample_target_column, 'patient_id', 'overall_survival_days', 'deceased'}
        if task == 'survival':
            exclude_cols.add(config_file_tree.TIME_COL)

        sample_feature_cols = [c for c in df_train_val_proc.columns if c not in exclude_cols]
        num_sample_features = len(sample_feature_cols)
        logging.info(f"Processed sample features ({num_sample_features}): {sample_feature_cols}")

        # update metadata_x in graphs loaded in RAM
        for i in train_idx:
            graph = train_val_dataset[i]
            feats = df_train_val_proc.loc[graph.sample_id, sample_feature_cols].values.astype(np.float32)
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_features]

        for i in val_idx:
            graph = train_val_dataset[i]
            feats = df_train_val_proc.loc[graph.sample_id, sample_feature_cols].values.astype(np.float32)
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

        for graph in test_dataset:
            feats = df_test_proc.loc[graph.sample_id, sample_feature_cols].values.astype(np.float32)
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


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
    """
    elif task == 'survival':
        model_samples_encoder = MultiOmicGATSurvival(
            in_node_features=num_genes_features,
            hidden_dim=64,
            out_channels=1,
            heads=2,
            dropout_rate=0.3,
            num_samples_features=num_sample_features if use_metadata else 0,
        ).to(device)
    """


    """"" fold specific dataset separation and DataLoader """""

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


    if config_file_tree.SAMPLES_LEVEL_TRAIN:
        logging.info(str(model_samples_encoder) + "\n")
        model_name = f"best_{task}_model_fold{fold + 1}.pt"

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

        for epoch in tqdm(range(1, epochs + 1)):
            if task == 'classification':
                train_loss, train_acc = train_epoch(device, model_samples_encoder, train_loader, optimizer_samples, criterion_samples)
                val_loss, val_metrics = evaluate(device, model_samples_encoder, val_loader, criterion_samples)

                scheduler.step(val_loss)  # update learning rate

                logging.info(f"[SAMPLES] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
                logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

                if val_loss < best_val_loss:
                    logging.info("IMPROVEMENT!\n")
                    best_val_loss = val_loss
                    torch.save(model_samples_encoder.state_dict(), os.path.join(output_dir, model_name))
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
                    torch.save(model_samples_encoder.state_dict(), os.path.join(output_dir, model_name))
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            if early_stopping_counter > 20:
                logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
                break

        best_model_path = os.path.join(output_dir, model_name)
        model_samples_encoder.load_state_dict(torch.load(best_model_path, map_location=device))

    else:
        logging.info(
            f"--- Loading Sample Encoder pre-trained model from {pretrained_sample_encoder_path} ---")

        # Gestione del percorso: supporta sia un file unico pre-addestrato sia file pre-addestrati specifici per fold
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
        logging.info('--- PATIENT-LEVEL TRAINING: MIL finetuning on patient target label ---')

        patient_clinical_map = {}
        num_patients_features = 0


        """"" patient features fold specific processing """""

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

            # saving preprocessors for evaluation/model_patients usage
            joblib.dump(
                {'encoders': p_encoders, 'scalers': p_scalers, 'imputers': p_imputers, 'categories': p_categories},
                os.path.join(output_dir, f"clinical_preprocessors_fold{fold + 1}.joblib")
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

        model_patient_encoder = PatientMILModel(
            sample_encoder=model_samples_encoder,  # passing samples pre-trained encoder
            num_classes=num_classes_patient_target,
            num_clinical_features=num_patients_features if use_clinical else 0
        ).to(device)
        logging.info(str(model_patient_encoder) + "\n")


        train_patients_targets = torch.tensor([patient_target_map[patient_id] for patient_id in patient_train_dataset.patient_ids],
                                              dtype=torch.long)

        class_counts_patients = torch.bincount(train_patients_targets, minlength=num_classes_patient_target)
        class_counts_patients = torch.clamp(class_counts_patients, min=1)  # TODO da lasciare? per prevenire divisioni a 0

        class_weights_patients = (len(train_patients_targets) / (len(class_counts_patients) * class_counts_patients.float())).to(device)
        criterion_patients = torch.nn.CrossEntropyLoss(weight=class_weights_patients)

        logging.info("Samples classes counts: " + str(class_counts_patients))
        logging.info("Sample classes weights: " + str(class_weights_patients))

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
        name_model_patient = f"best_patients_model_fold{fold + 1}.pt"
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in tqdm(range(1, epochs + 1)):
            train_loss, train_acc = train_epoch_mil(device, model_patient_encoder, patients_train_loader, optimizer_patients, criterion_patients)
            val_loss, val_metrics = evaluate_mil(device, model_patient_encoder, patients_val_loader, criterion_patients)

            logging.info(f"[PATIENTS] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
            logging.info(f"Val metrics: Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}"
            )

            if val_loss < best_val_loss:
                logging.info("PATIENT MODEL IMPROVEMENT!\n")
                best_val_loss = val_loss
                torch.save(model_patient_encoder.state_dict(), os.path.join(output_dir, name_model_patient))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter > 15:
                logging.info(f"Patients model Early Stopping triggered at epoch {epoch}\n")
                break

        # Test Evaluation on best patient encoder model
        model_patient_encoder.load_state_dict(torch.load(os.path.join(output_dir, name_model_patient), map_location=device))

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
show_fold_results(pd.DataFrame(samples_fold_results), task, save_to_csv=True, dir_to_save=output_dir)

if config_file_tree.PATIENTS_LEVEL_TRAIN and patients_fold_results:
    logging.info("--- PATIENT LEVEL RESULTS ---")
    show_fold_results(pd.DataFrame(patients_fold_results), task, save_to_csv=True, dir_to_save=output_dir)

logging.info("DONE\n")