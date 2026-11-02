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


def train_epoch_mil(device, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total_patients = 0

    optimizer.zero_grad()

    for patient_batch in loader:
        patient_batch = patient_batch.to(device)

        # Il target del paziente è memorizzato in patient_batch.patient_y
        target_y = patient_batch.patient_y

        logits, _ = model(patient_batch)
        loss = criterion(logits, target_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * target_y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target_y).sum().item()
        total_patients += target_y.size(0)

    return total_loss / total_patients, correct / total_patients


def evaluate_mil(device, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_patients = 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for patient_batch in loader:
            patient_batch = patient_batch.to(device)
            target_y = patient_batch.patient_y  # Target paziente estratto direttamente dal batch PyG

            logits, _ = model(patient_batch)
            loss = criterion(logits, target_y)

            # Accumula la loss pesata sul numero di pazienti nel batch
            total_loss += loss.item() * target_y.size(0)
            total_patients += target_y.size(0)

            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            all_targets.extend(target_y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / total_patients if total_patients > 0 else 0.0

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


def evaluate_mil_with_attention(device, model, loader, criterion):
    """Valuta il modello MIL ed estrae i pesi d'attenzione associati al corretto paziente."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    model.eval()
    total_loss = 0.0
    total_patients = 0
    all_targets, all_preds, all_probs = [], [], []
    attention_records = []

    with torch.no_grad():
        for item in loader:
            # Unpacking del batch
            if isinstance(item, (list, tuple)):
                patient_batch = item[0].to(device)
                target_y = item[1].to(device)
                raw_p_ids = item[2] if len(item) > 2 else getattr(patient_batch, 'patient_id', "unknown")
            else:
                patient_batch = item.to(device)
                target_y = patient_batch.patient_y
                raw_p_ids = getattr(patient_batch, 'patient_id', "unknown")

            logits, attn_weights = model(patient_batch)
            loss = criterion(logits, target_y)

            total_loss += loss.item() * target_y.size(0)
            total_patients += target_y.size(0)

            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            all_targets.extend(target_y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Converti i pesi d'attenzione in un array 1D NumPy
            weights = np.atleast_1d(attn_weights.squeeze().cpu().numpy())

            # Formattazione degli ID dei campioni
            if hasattr(patient_batch, 'sample_id') and patient_batch.sample_id is not None:
                raw_sids = patient_batch.sample_id
                if isinstance(raw_sids, (list, tuple)) and len(raw_sids) > 0 and isinstance(raw_sids[0], (list, tuple)):
                    sample_ids = [s for sublist in raw_sids for s in sublist]
                elif isinstance(raw_sids, (list, tuple)):
                    sample_ids = list(raw_sids)
                elif isinstance(raw_sids, np.ndarray):
                    sample_ids = raw_sids.flatten().tolist()
                else:
                    sample_ids = [raw_sids]
            else:
                sample_ids = [f"sample_{j}" for j in range(len(weights))]

            # Formattazione delle etichette dei tessuti
            if hasattr(patient_batch, 'y') and patient_batch.y is not None:
                tissue_labels = patient_batch.y.cpu().numpy().tolist()
            else:
                tissue_labels = [None] * len(weights)

            # -------------------------------------------------------------------
            # ASSEGNAZIONE CORRETTA DEL PATIENT_ID A CIASCUN CAMPIONE
            # -------------------------------------------------------------------
            if isinstance(raw_p_ids, (list, tuple)):
                p_id_list = [str(p[0] if isinstance(p, (list, tuple)) else p).strip() for p in raw_p_ids]
            else:
                p_id_list = [str(raw_p_ids).strip()]

            sample_patient_ids = []

            # 1. Mappatura tramite num_samples per ciascun paziente nel batch
            if hasattr(patient_batch, 'num_samples') and patient_batch.num_samples is not None:
                num_samples_list = patient_batch.num_samples
                if hasattr(num_samples_list, 'tolist'):
                    num_samples_list = num_samples_list.tolist()

                for p_idx, n_s in enumerate(num_samples_list):
                    curr_pid = p_id_list[p_idx] if p_idx < len(p_id_list) else p_id_list[0]
                    sample_patient_ids.extend([curr_pid] * n_s)

            # 2. Fallback: estrazione dal prefisso dello sample_id (es. 'H147_pAsc1_RNA2' -> 'H147')
            if len(sample_patient_ids) != len(sample_ids):
                sample_patient_ids = [str(s_id).split('_')[0] for s_id in sample_ids]

            # Associazione 1-a-1 priva di disallineamenti
            for s_id, p_id_curr, w, t_lab in zip(sample_ids, sample_patient_ids, weights, tissue_labels):
                attention_records.append({
                    'patient_id': p_id_curr,
                    'sample_id': s_id,
                    'tissue_label': t_lab,
                    'attention_weight': float(w)
                })

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / total_patients if total_patients > 0 else 0.0

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

    df_attention = pd.DataFrame(attention_records)
    return avg_loss, metrics, df_attention


""""" results """""

def show_fold_results(df_fold_results, task, save_to_csv = False, dir_to_save=None, filename=None):
    """
        Prints metrics summary (Mean ± Var) to logs and appends summary statistics
        rows (Mean ± Var, mean, var, std) as final rows in the output TSV/CSV report.
        """
    from os import path
    import pandas as pd
    import logging

    numeric_cols = [c for c in df_fold_results.columns if c != 'fold']
    mean_series = df_fold_results[numeric_cols].mean()
    var_series = df_fold_results[numeric_cols].var()

    logging.info("--- OVERALL FOLD STATISTICS (MEAN ± VAR) ---")
    for col in numeric_cols:
        logging.info(f"{col}: {mean_series[col]:.4f} ± {var_series[col]:.4f} (Var)")

    if save_to_csv and dir_to_save:
        df_save = df_fold_results.copy()

        # Format row with mean +- variance strings
        summary_row = {'fold': 'Mean ± Var'}
        for c in numeric_cols:
            summary_row[c] = f"{mean_series[c]:.4f} ± {var_series[c]:.4f}"

        mean_row = {'fold': 'mean', **mean_series.to_dict()}
        var_row = {'fold': 'var', **var_series.to_dict()}
        std_row = {'fold': 'std', **df_fold_results[numeric_cols].std().to_dict()}

        # Append overall statistics rows to the final dataframe
        df_save = pd.concat([df_save, pd.DataFrame([summary_row, mean_row, var_row, std_row])], ignore_index=True)

        out_name = filename if filename else f"report_kfold_test_results_{task}.csv"
        out_path = path.join(dir_to_save, out_name)
        df_save.to_csv(out_path, index=False)
        logging.info(f"Final folds report with summary statistics saved to: {out_path}\n")


def save_learning_curves(history, fold_num, output_dir="learning_curves", is_grid_search=False):
    """
    Plots Train and Val Loss/Metrics curves.

    Parameters:
    - history: dict with lists ['train_loss', 'val_loss', 'train_metric', 'val_metric']
    - fold_num: fold index (es. 1, 2, ...)
    - output_dir: root folder
    """
    import matplotlib.pyplot as plt
    from os import makedirs, path

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- LOSS TRAJECTORY ---
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', linestyle='--', linewidth=2)
    if is_grid_search:
        ax1.set_title(f'Cfg {fold_num} - Loss Trajectory')
    else:
        ax1.set_title(f'Fold {fold_num} - Loss Trajectory')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- METRIC TRAJECTORY ---
    metric_label = 'Accuracy'

    if any(m > 0 for m in history['train_metric']):
        ax2.plot(epochs, history['train_metric'], label=f'Train {metric_label}', color='#2ca02c', linewidth=2)

    ax2.plot(epochs, history['val_metric'], label=f'Val {metric_label}', color='#ff7f0e', linestyle='--', linewidth=2)
    if is_grid_search:
        ax2.set_title(f'Cfg {fold_num} - {metric_label} Performance')
    else:
        ax2.set_title(f'Fold {fold_num} - {metric_label} Performance')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric_label)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if is_grid_search:
        save_dir = path.join(output_dir)
        makedirs(save_dir, exist_ok=True)
        save_path = path.join(save_dir, f'cfg{fold_num}.png')
    else:
        save_dir = path.join(output_dir)
        makedirs(save_dir, exist_ok=True)
        save_path = path.join(save_dir, f'learning_curve_fold{fold_num}.png')

    plt.savefig(save_path, dpi=300)
    plt.close()


def set_sample_encoder_freezing(model, freeze: bool = True):
    """
    Attiva o disattiva il congelamento dei parametri dello Sample Encoder (GAT).
    """
    for param in model.sample_encoder.parameters():
        param.requires_grad = not freeze

    if freeze:
        model.sample_encoder.eval()
        logging.info(" [INFO] Sample Encoder (GAT) frozen. Training only MIL head.")
    else:
        model.sample_encoder.train()
        logging.info(" [INFO] Sample Encoder (GAT) unfrozen. Fine-tuning on.")


def apply_permutation_test(patient_dataset, seed=42):
    """
    Mantiene inalterate le bag e i grafi dei pazienti,
    ma mescola in modo casuale le etichette di resistenza al platino.
    """
    np.random.seed(seed)

    # Estrazione delle etichette dei pazienti
    original_labels = np.array([bag.patient_y.item() for bag in patient_dataset])

    # Permutazione casuale dei target
    permuted_labels = np.random.permutation(original_labels)

    # Assegnazione delle etichette mescolate al dataset
    for idx, bag in enumerate(patient_dataset):
        bag.patient_y = torch.tensor(permuted_labels[idx], dtype=torch.long)

    print(f"[PERMUTATION TEST] Etichette mescolate con successo. "
          f"Distribuzione originale: {np.bincount(original_labels)} | "
          f"Distribuzione permutata: {np.bincount(permuted_labels)}")

    return patient_dataset


def save_patient_classification_results(
        device,
        model,
        test_loader,
        label_encoder_patient,
        fold_num,
        output_dir,
        label_encoder_sample=None,
        clinical_file_path=None,
        clinical_features_to_include=None
):
    """
    Valuta il modello MIL sui pazienti di test, unisce le feature cliniche desiderate
    e salva due file TSV per il fold:
    - fold_{fold_num}_correct_patients.tsv
    - fold_{fold_num}_misclassified_patients.tsv
    """
    import os
    import pandas as pd
    import config_file_tree
    model.eval()

    def _unwrap_id(val):
        while isinstance(val, (list, tuple)):
            if len(val) > 0:
                val = val[0]
            else:
                return ""
        if hasattr(val, 'item'):
            val = val.item()
        return str(val).strip()

    predictions_dir = os.path.join(output_dir, "patient_predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    records = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            targets = batch.patient_y if hasattr(batch, 'patient_y') else batch.y

            if hasattr(batch, 'patient_id'):
                raw_ids = batch.patient_id
            else:
                raw_ids = [f"Patient_{i}" for i in range(len(targets))]

            # Estrazione dei tissue_types dal batch
            # Se batch contiene y (etichetta del tessuto) o tissue_type
            batch_tissue_names = []
            if hasattr(batch, 'y') and batch.y is not None:
                y_tissues = batch.y.cpu().numpy()
                if label_encoder_sample is not None:
                    batch_tissue_names = [label_encoder_sample.classes_[t] for t in y_tissues]
                else:
                    batch_tissue_names = [str(t) for t in y_tissues]
            elif hasattr(batch, 'tissue_type'):
                batch_tissue_names = [str(t) for t in batch.tissue_type]

            # Mappatura dei tessuti al rispettivo paziente tramite bag_index
            if hasattr(batch, 'num_samples') and batch.num_samples is not None:
                num_samples_per_patient = batch.num_samples
            else:
                num_samples_per_patient = [len(batch_tissue_names)] if len(targets) == 1 else []

            sample_offset = 0
            for idx in range(len(targets)):
                item_raw = raw_ids[idx] if idx < len(raw_ids) else f"Patient_{idx}"
                p_id = _unwrap_id(item_raw)

                y_true = targets[idx].item()
                y_pred = preds[idx].item()
                p_0 = probs[idx][0].item()
                p_1 = probs[idx][1].item() if probs.shape[1] > 1 else (1.0 - p_0)

                target_name = label_encoder_patient.classes_[y_true] if label_encoder_patient is not None else str(
                    y_true)
                pred_name = label_encoder_patient.classes_[y_pred] if label_encoder_patient is not None else str(y_pred)

                # Estrazione della lista unica dei tipi di tessuto del paziente
                if sample_offset < len(batch_tissue_names) and idx < len(num_samples_per_patient):
                    n_samples = num_samples_per_patient[idx]
                    patient_tissues = batch_tissue_names[sample_offset: sample_offset + n_samples]
                    sample_offset += n_samples
                    unique_tissues_str = ", ".join(sorted(list(set(patient_tissues))))
                else:
                    unique_tissues_str = "Unknown"

                records.append({
                    'patient_id': p_id,
                    'target_label': y_true,
                    'target_label_name': target_name,
                    'predicted_label': y_pred,
                    'predicted_label_name': pred_name,
                    'prob_class_0': round(p_0, 4),
                    'prob_class_1': round(p_1, 4),
                    'tissue_types': unique_tissues_str,
                    'is_correct': (y_true == y_pred)
                })

    df_results = pd.DataFrame(records)
    df_results['patient_id'] = df_results['patient_id'].astype(str)

    # -------------------------------------------------------------------
    # RECUPERO E MERGE DELLE FEATURE CLINICHE
    # -------------------------------------------------------------------
    if clinical_file_path is None:
        clinical_file_path = os.path.join(
            config_file_tree.PREPROCESSED_DATA_DIR,
            "preprocessed_clinical_data.tsv"
        )

    if os.path.exists(clinical_file_path):
        df_clinical_full = pd.read_csv(clinical_file_path, sep='\t')

        if 'patient_id' in df_clinical_full.columns:
            df_clinical_full['patient_id'] = df_clinical_full['patient_id'].astype(str).str.strip()

            if clinical_features_to_include is not None:
                valid_cols = ['patient_id'] + [c for c in clinical_features_to_include if c in df_clinical_full.columns]
                df_clinical_subset = df_clinical_full[valid_cols].drop_duplicates(subset=['patient_id'])
            else:
                df_clinical_subset = df_clinical_full.drop_duplicates(subset=['patient_id'])

            df_results = df_results.merge(df_clinical_subset, on='patient_id', how='left')
        else:
            print(f"[WARNING] Colonna 'patient_id' non trovata in {clinical_file_path}")
    else:
        print(f"[WARNING] File clinico non trovato in {clinical_file_path}")

    # Separazione predizioni corrette ed errate
    df_correct = df_results[df_results['is_correct'] == True].drop(columns=['is_correct'])
    df_misclassified = df_results[df_results['is_correct'] == False].drop(columns=['is_correct'])

    correct_path = os.path.join(predictions_dir, f"fold_{fold_num}_correct_patients.tsv")
    misclassified_path = os.path.join(predictions_dir, f"fold_{fold_num}_misclassified_patients.tsv")

    df_correct.to_csv(correct_path, sep='\t', index=False)
    df_misclassified.to_csv(misclassified_path, sep='\t', index=False)

    print(f"[Fold {fold_num}] Salvati TSV predizioni pazienti:")
    print(f"  - Corretti    ({len(df_correct)} pazienti): {correct_path}")
    print(f"  - Errati      ({len(df_misclassified)} pazienti): {misclassified_path}")

    return df_correct, df_misclassified