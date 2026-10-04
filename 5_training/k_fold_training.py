import logging
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from models.GAT import GAT
from train_eval_utils import print_details, pre_train_imputation_encoding
from SampleGraphDataset import PatientTissueDataset
from models.MultiOmicGATSurvival import MultiOmicGATSurvival, CoxPHLoss
from train_functions import train_epoch, evaluate, train_epoch_survival, evaluate_survival


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

task = config_file_tree.TASK
target_column = config_file_tree.TARGET_COL

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")

path_train_val_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_train_val.csv")
path_test_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")

path_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR

dir_to_save = os.path.join("trained_models")
os.makedirs(dir_to_save, exist_ok=True)


""""" loading clinical data_HERCULES_DECIDER """""

df_metadata = pd.read_csv(path_to_metadata, sep="\t")

if use_metadata:
    df_metadata = df_metadata[['sample_id', 'patient_id', target_column] + config_file_tree.METADATA_FEATURES]

metadata_train_val_samples = pd.read_csv(path_train_val_csv)
metadata_test_samples = pd.read_csv(path_test_csv)

df_metadata_train_val = df_metadata[
    df_metadata['sample_id'].isin(metadata_train_val_samples['sample_id'])
].dropna(subset=[target_column]).copy()

df_metadata_test = df_metadata[
    df_metadata['sample_id'].isin(metadata_test_samples['sample_id'])
].dropna(subset=[target_column]).copy()


""""" target encoding  """""

if task == 'classification':
    logging.info(f"Starting training for classification task using target feature: {target_column}\n")

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(df_metadata[target_column].dropna())

    df_metadata_train_val[target_column] = label_encoder.transform(df_metadata_train_val[target_column])
    df_metadata_test[target_column] = label_encoder.transform(df_metadata_test[target_column])

    joblib.dump(label_encoder, os.path.join(dir_to_save, f"label_encoder_{target_column}.joblib"))
    logging.info("Classes identified:\n" + str(label_encoder.classes_))
else:
    logging.info(f"Starting training on survival task using target event: {target_column}\n")

#TODO mi sa che non cambia nulla ma su RAM è meglio apparently
'''
logging.info("Train + Val Dataset init...")
train_val_dataset = PatientTissueDataset(path_graphs_dir, df_metadata_train_val, task=task, target_col=target_column)

logging.info("Test Dataset init...")
test_dataset = PatientTissueDataset(path_graphs_dir, df_metadata_test, task=task, target_col=target_column)
'''
# --- IN-MEMORY CACHING (Pre-caricamento in RAM) ---
logging.info("Train + Val Dataset init...")
raw_train_val_dataset = PatientTissueDataset(path_graphs_dir, df_metadata_train_val, task=task, target_col=target_column)

logging.info("Test Dataset init...")
raw_test_dataset = PatientTissueDataset(path_graphs_dir, df_metadata_test, task=task, target_col=target_column)

logging.info("Pre-loading datasets into RAM to avoid disk I/O bottleneck...")
train_val_dataset = [raw_train_val_dataset[i] for i in tqdm(range(len(raw_train_val_dataset)), desc="Loading Train/Val")]
test_dataset = [raw_test_dataset[i] for i in tqdm(range(len(raw_test_dataset)), desc="Loading Test")]

print_details(raw_train_val_dataset)


""""" set model parameters """""

num_genes_features = raw_train_val_dataset.num_features  # multiomics feature vector
num_sample_features = len(raw_train_val_dataset.sample_feature_cols)  # total sample features without the one to predict
logging.info(f"Sample Features found : {num_sample_features} - {raw_train_val_dataset.sample_feature_cols}")  # TODO fix print/locazione

epochs = 100


""""" k-fold train loop """""

criterion = None
y_skf = None

if task == 'classification':
    #y_labels = torch.tensor([train_val_dataset[i].y.item() for i in range(len(train_val_dataset))])
    y_labels = torch.tensor(df_metadata_train_val[target_column].values, dtype=torch.long)
    y_skf = y_labels
    unique_labels = torch.unique(y_labels)
    logging.info(f"Unique labels found in dataset: {unique_labels}")
    num_classes = len(unique_labels)

    # different weights to classes based on number of samples

    class_counts = torch.bincount(y_labels)
    total_samples = len(y_labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    logging.info("classes_counts: " + str(class_counts))

    logging.info("weights: " + str(class_weights))

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

elif task == 'survival':
    # stratification based on event (0 = censured, 1 = event)
    #y_events = torch.tensor([train_val_dataset[i].y_event.item() for i in range(len(train_val_dataset))])
    y_events = torch.tensor(df_metadata_train_val[target_column].values, dtype=torch.float32)
    y_skf = y_events

    criterion = CoxPHLoss()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset, y_skf)):
    logging.info(f"\n--- FOLD {fold + 1} ---\n")

    model_name = f"best_{task}_model_fold{fold + 1}.pt"

    """"" target and metadata fold specific processing """""

    if use_metadata:
        logging.info("Executing fold-specific feature/target encoding and preprocessing...")

        # retrieving patient ids of current train/val splits from graphs sample ids
        df_metadata_train_raw = df_metadata_train_val.iloc[train_idx].copy()
        df_metadata_val_raw = df_metadata_train_val.iloc[val_idx].copy()
        df_metadata_test_raw = df_metadata_test.copy()

        # fit-transform only on train set
        df_train_proc, encoders, scalers, imputers, categories = pre_train_imputation_encoding(
            df=df_metadata_train_raw,
            task=task,
            target_col=target_column,
            is_train=True
        )

        # transform of validation and test
        df_val_proc = pre_train_imputation_encoding(
            df=df_metadata_val_raw,
            task=task,
            target_col=target_column,
            is_train=False,
            encoders=encoders,
            scalers=scalers,
            imputers=imputers,
            categories=categories
        )

        df_test_proc = pre_train_imputation_encoding(
            df=df_metadata_test_raw,
            task=task,
            target_col=target_column,
            is_train=False,
            encoders=encoders,
            scalers=scalers,
            imputers=imputers,
            categories=categories
        )

        # saving preprocessors for evaluation/model usage
        preprocessors = {
            'encoders': encoders,
            'scalers': scalers,
            'imputers': imputers,
            'categories': categories
        }
        prep_path = os.path.join(dir_to_save, f"clinical_preprocessors_{task}_fold{fold + 1}.joblib")
        joblib.dump(preprocessors, prep_path)

        # 3. Unione di Train e Val elaborati e posizionamento di 'sample_id' come Indice
        df_train_val_proc = pd.concat([df_train_proc, df_val_proc], axis=0)

        if 'sample_id' in df_train_val_proc.columns:
            df_train_val_proc = df_train_val_proc.set_index('sample_id')
        if 'sample_id' in df_test_proc.columns:
            df_test_proc = df_test_proc.set_index('sample_id')

        # 4. Selezione delle sole colonne di feature numeriche elaborate
        exclude_cols = {target_column, 'patient_id', 'overall_survival_days', 'deceased'}
        if task == 'survival' and hasattr(config_file_tree, 'TIME_COL'):
            exclude_cols.add(config_file_tree.TIME_COL)

        sample_feature_cols = [c for c in df_train_val_proc.columns if c not in exclude_cols]
        num_sample_features = len(sample_feature_cols)
        logging.info(f"Number of processed sample features for Fold {fold + 1}: {num_sample_features}")

        # 7. Aggiornamento dinamico del campo metadata_x nei grafi in RAM
        # Aggiornamento grafi Train
        for i in train_idx:
            graph = train_val_dataset[i]
            feats = df_train_val_proc.loc[graph.sample_id, sample_feature_cols].values.astype(np.float32)
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_features]

        # Aggiornamento grafi Validation
        for i in val_idx:
            graph = train_val_dataset[i]
            feats = df_train_val_proc.loc[graph.sample_id, sample_feature_cols].values.astype(np.float32)
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

        # Aggiornamento grafi Test
        for graph in test_dataset:
            feats = df_test_proc.loc[graph.sample_id, sample_feature_cols].values.astype(np.float32)
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


    """"" fold specific dataset separation and DataLoader """""

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


    """"" task specific model upload """""

    if task == 'classification': # TODO per ora no edge features

        model = GAT(
            num_node_features=num_genes_features,
            num_classes=num_classes,
            hidden_channels=64,
            heads=2,
        ).to(device)
        '''
        model = MultiOmicGAT(
            in_node_features=num_genes_features,
            hidden_dim=64,
            out_channels=num_classes,
            heads=2,
            dropout=0.3,
            num_sample_features=num_sample_features if use_metadata else 0,
        ).to(device)
        '''
    elif task == 'survival':
        model = MultiOmicGATSurvival(
            in_node_features=num_genes_features,
            hidden_dim=64,
            out_channels=1,
            heads=2,
            dropout_rate=0.3,
            num_samples_features=num_sample_features if use_metadata else 0,
        ).to(device)

    logging.info(str(model) + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    if task == 'classification':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif task == 'survival':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


    """"" fold train loop """""

    # possible metrics to evaluate for improvement
    best_val_loss = float('inf')  # minimizing loss
    best_val_c_index = 0.0  # maximizing C-index
    early_stopping_counter = 0

    for epoch in tqdm(range(1, epochs + 1)):
        if task == 'classification':
            train_loss, train_acc = train_epoch(device, model, train_loader, optimizer, criterion)
            val_loss, val_metrics = evaluate(device, model, val_loader, criterion)

            scheduler.step(val_loss)  # update learning rate

            logging.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
            logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

            if val_loss < best_val_loss:
                logging.info("Loss IMPROVEMENT!\n")
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(dir_to_save, model_name))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        elif task == 'survival':
            train_loss = train_epoch_survival(device, model, train_loader, optimizer, criterion)
            val_loss, val_metrics = evaluate_survival(device, model, val_loader, criterion)
            val_c_index = val_metrics['c_index']

            scheduler.step(val_c_index)  # update learning rate

            logging.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}"
            )

            if val_c_index > best_val_c_index:
                logging.info(
                    f"--> C-Index IMPROVEMENT!: {best_val_c_index:.4f} -> {val_c_index:.4f}\n")
                best_val_c_index = val_c_index
                torch.save(model.state_dict(), os.path.join(dir_to_save, model_name))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
            break

    best_model_path = os.path.join(dir_to_save, model_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if task == 'classification':
        test_loss, test_metrics = evaluate(device, model, test_loader, criterion)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_metrics['acc']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test AUC: {test_metrics['auc']:.4f}\n"
        )
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })

    elif task == 'survival':
        test_loss, test_metrics = evaluate_survival(device, model, test_loader, criterion)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test C-Index: {test_metrics['c_index']:.4f}\n"
        )
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'c_index': test_metrics['c_index']
        })


""""" folds aggregate evaluation """""

df_fold_results = pd.DataFrame(fold_results)

if task == 'classification':
    avg_acc = df_fold_results['acc'].mean()
    std_acc = df_fold_results['acc'].std()
    avg_f1 = df_fold_results['f1'].mean()
    std_f1 = df_fold_results['f1'].std()
    avg_auc = df_fold_results['auc'].mean()
    std_auc = df_fold_results['auc'].std()

    logging.info(f"Accuracy:  {avg_acc:.4f} ± {std_acc:.4f}")
    logging.info(f"F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    logging.info(f"ROC AUC:   {avg_auc:.4f} ± {std_auc:.4f}")

elif task == 'survival':
    avg_c_index = df_fold_results['c_index'].mean()
    std_c_index = df_fold_results['c_index'].std()

    logging.info(f"C-Index:   {avg_c_index:.4f} ± {std_c_index:.4f}")

df_fold_results.to_csv(os.path.join(dir_to_save, f"report_kfold_test_results_{task}.csv"), index=False)  # final report saved

logging.info("DONE\n")