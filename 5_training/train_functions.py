""""" functions """""
import logging

import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log', mode='w'),
        logging.StreamHandler()
    ]
)

""""" for classification task """""

def train_epoch(device, model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data in loader:  # data_HERCULES_DECIDER = batch of graphs
        data = data.to(device)
        optimizer.zero_grad()
        metadata_x = getattr(data, 'metadata_x', None)
        out, _ = model(data.x, data.edge_index, data.batch, metadata_x=metadata_x)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        total += data.num_graphs

    return total_loss / total, correct / total


def evaluate(device, model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            metadata_x = getattr(data, 'metadata_x', None)
            out, _ = model(data.x, data.edge_index, data.batch, metadata_x=metadata_x)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)  # class with the highest prob
            correct += (pred == data.y).sum().item()

            all_targets.extend(data.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += data.num_graphs

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    num_classes_in_target = len(np.unique(all_targets))
    if num_classes_in_target == 2:
        auc_val = roc_auc_score(all_targets, all_probs[:, 1])
    elif num_classes_in_target > 2:  # Multi-Class AUC --> OvR strategy (One-vs-Rest)
        auc_val = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    metrics = {
        'acc': np.mean(all_preds == all_targets),  # acc = correct/total
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'auc': auc_val,
    }

    return total_loss/total, metrics


""""" for survival task """""

def train_epoch_survival(device, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # output = Log-Hazard (relative risk)
        metadata_x = getattr(data, 'metadata_x', None)
        risk_pred, _ = model(data.x, data.edge_index, data.batch, metadata_x=metadata_x)

        # Cox Loss based on (risk_pred, event, time)
        loss = criterion(risk_pred, data.y_event, data.y_time)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    return total_loss / total_samples


def evaluate_survival(device, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_risk_scores = []
    all_times = []
    all_events = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            metadata_x = getattr(data, 'metadata_x', None)
            risk_pred, _ = model(data.x, data.edge_index, data.batch, metadata_x=metadata_x)

            loss = criterion(risk_pred, data.y_event, data.y_time)
            total_loss += loss.item() * data.num_graphs

            all_risk_scores.extend(risk_pred.squeeze(-1).cpu().numpy())
            all_times.extend(data.y_time.cpu().numpy())
            all_events.extend(data.y_event.cpu().numpy())

    all_risk_scores = np.array(all_risk_scores)
    all_times = np.array(all_times)
    all_events = np.array(all_events)

    # higher risk_score means higher risk (lower survival time), we pass the opposite (-risk_scores) to concordance_index
    c_index = concordance_index(
        event_times=all_times,
        predicted_scores=-all_risk_scores,
        event_observed=all_events
    )

    metrics = {
        'c_index': c_index
    }

    return total_loss / len(all_times), metrics


""""" for MIL patient classification task """""

def train_epoch_mil(device, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total_patients = 0

    for patient_batch, target_y, _ in loader:
        patient_batch = patient_batch.to(device)
        target_y = target_y.to(device)

        optimizer.zero_grad()
        logits, _ = model(patient_batch)

        loss = criterion(logits, target_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target_y).sum().item()
        total_patients += target_y.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = correct / total_patients if total_patients > 0 else 0.0
    return avg_loss, acc


def evaluate_mil(device, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for patient_batch, target_y, _ in loader:
            patient_batch = patient_batch.to(device)
            target_y = target_y.to(device)

            logits, _ = model(patient_batch)
            loss = criterion(logits, target_y)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            all_targets.extend(target_y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    num_classes_in_target = len(np.unique(all_targets))
    if num_classes_in_target == 2:
        auc_val = roc_auc_score(all_targets, all_probs[:, 1])
    elif num_classes_in_target > 2:
        auc_val = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    metrics = {
        'acc': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'auc': auc_val
    }

    return avg_loss, metrics


""""" results """""

def show_fold_results(df_fold_results, task, save_to_csv = False, dir_to_save=None):
    from os import path
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

    if save_to_csv:
        df_fold_results.to_csv(path.join(dir_to_save, f"report_kfold_test_results_{task}.csv"),index=False)
        logging.info("\nFinal folds report saved!\n")
