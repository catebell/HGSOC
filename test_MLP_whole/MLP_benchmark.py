import copy
import os
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

import config_file_tree

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlp_benchmark.log', mode='w'),
        logging.StreamHandler()
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    cpus_allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    try:
        torch.set_num_threads(cpus_allocated)
        torch.set_num_interop_threads(max(1, cpus_allocated // 2))
    except Exception as e:
        pass


def save_benchmark_visualizations(df_results, output_dir='results_visualizations'):
    """
    Genera e salva per ciascuna feature target:
    1. Una tabella elegante formattata in PNG.
    2. Una heatmap grafica (sulla metrica ROC-AUC) che incrocia Tissue_Mode e Modality.
    """
    os.makedirs(output_dir, exist_ok=True)
    targets = df_results['Target'].unique()

    for target in targets:
        df_target = df_results[df_results['Target'] == target].copy()

        # ==============================================================================
        # 1. SALVATAGGIO TABELLA PNG
        # ==============================================================================
        display_cols = ['Tissue_Mode', 'Modality', 'Accuracy', 'F1-Macro', 'ROC-AUC']
        col_labels = ['Tissue Mode', 'Modality', 'Accuracy', 'F1-Macro', 'ROC-AUC']
        table_data = df_target[display_cols].values

        fig, ax = plt.subplots(figsize=(10, 2.5 + 0.45 * len(df_target)))
        ax.axis('off')

        table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Styling visivo della tabella
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1B365D')  # Blu scuro intestazione
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
            else:
                cell.set_facecolor('#F8F9FA' if r % 2 == 0 else '#FFFFFF')  # Righe alternate

        plt.title(f"MLP Benchmark Results - Target: {target}", fontsize=13, pad=15, weight='bold')
        plt.tight_layout()

        table_path = os.path.join(output_dir, f'summary_table_{target}.png')
        plt.savefig(table_path, bbox_inches='tight', dpi=300)
        plt.close()

        # ==============================================================================
        # 2. SALVATAGGIO HEATMAP PNG (Matrice Tissue Mode vs Modality su ROC-AUC)
        # ==============================================================================
        # Estrazione del valore medio numerico da "0.6887 ± 0.0539" per colorare la mappa
        df_target['ROC-AUC_mean'] = df_target['ROC-AUC'].apply(
            lambda x: float(x.split('±')[0].strip()) if isinstance(x, str) and '±' in x else float(x)
        )

        pivot_auc = df_target.pivot(index='Tissue_Mode', columns='Modality', values='ROC-AUC_mean')
        pivot_text = df_target.pivot(index='Tissue_Mode', columns='Modality', values='ROC-AUC')

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            pivot_auc,
            annot=pivot_text.values,
            fmt='',
            cmap='YlGnBu',
            cbar=True,
            ax=ax,
            linewidths=1.5,
            linecolor='gray',
            annot_kws={"size": 10, "weight": "bold"}
        )

        plt.title(f"ROC-AUC Heatmap - Target: {target}", fontsize=13, weight='bold', pad=12)
        plt.xlabel("Omic Modality", fontsize=10, weight='bold')
        plt.ylabel("Tissue Mode", fontsize=10, weight='bold')
        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, f'heatmap_roc_auc_{target}.png')
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
        plt.close()

        logging.info(f"Visualizzazioni salvate per '{target}': {table_path} e {heatmap_path}")


def save_learning_curves(history, modality, target_feat, fold_num, output_dir='learning_curves', task_name='classification'):
    """
    Genera e salva un grafico con le curve di Train e Validation Loss/Metriche.
    history: dict contenente le liste ['train_loss', 'val_loss', 'train_metric', 'val_metric']
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- GRAFICO 1: LOSS TRAJECTORY ---
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', linestyle='--', linewidth=2)
    ax1.set_title(f'Fold {fold_num} - Loss (Overfitting Check)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- GRAFICO 2: METRIC TRAJECTORY ---
    metric_label = 'Accuracy' if task_name == 'classification' else 'C-Index'
    ax2.plot(epochs, history['train_metric'], label=f'Train {metric_label}', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, history['val_metric'], label=f'Val {metric_label}', color='#ff7f0e', linestyle='--', linewidth=2)
    ax2.set_title(f'Fold {fold_num} - {metric_label} Performance')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric_label)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, target_feat, modality), exist_ok=True)
    save_path = os.path.join(output_dir, target_feat, modality, f'learning_curve_fold{fold_num}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==============================================================================
# 1. MODELLO MLP PER FEATURE OMICHE FLATTENED
# ==============================================================================

class TabularGeneMLP(nn.Module):
    """ MLP con Batch Normalization, Dropout e connessioni di proiezione per feature geniche """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.5):
        super(TabularGeneMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# 2. DATASET IN-MEMORY PER TABULAR DATA
# ==============================================================================

class InMemoryOmicDataset(Dataset):
    def __init__(self, features, labels, sample_ids, patient_ids):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_ids = sample_ids
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==============================================================================
# 3. SELEZIONE TESSUTI E CARICAMENTO GRAFI IN MEMORIA
# ==============================================================================

def select_most_fitting_sample(group_df):
    """
    Seleziona l'unico campione 'più rappresentativo' per paziente:
    1. Primario (phase_label == 'Primary')
    2. Non trattato (treatment == 'Untreated')
    3. Solido (is_fluid == 'no')
    """
    df = group_df.copy()

    # Priority score
    df['score'] = 0
    if 'phase_label' in df.columns:
        df['score'] += (df['phase_label'].astype(str).str.lower() == 'primary').astype(int) * 4
    if 'treatment' in df.columns:
        df['score'] += (df['treatment'].astype(str).str.lower() == 'untreated').astype(int) * 2
    if 'is_fluid' in df.columns:
        df['score'] += (df['is_fluid'].astype(str).str.lower() == 'no').astype(int) * 1

    df_sorted = df.sort_values(by=['score', 'sample_id'], ascending=[False, True])
    return df_sorted.iloc[0]


def load_raw_omic_data(graphs_dir, metadata_df, clinical_df, target_col, tissue_mode='all_tissues'):
    """
    Carica i grafi .pt, unisce i target dal file clinico ed estrae le matrici x [N_genes, 2].
    """
    # Key di join
    clinical_patient_col = clinical_df.columns[0]

    # Join metadata e clinical
    clinical_sub = clinical_df[[clinical_patient_col, target_col]].dropna(subset=[target_col]).copy()
    clinical_sub[clinical_patient_col] = clinical_sub[clinical_patient_col].astype(str).str.upper()

    metadata_df['patient_id_upper'] = metadata_df['patient_id'].astype(str).str.upper()
    merged_df = metadata_df.merge(clinical_sub, left_on='patient_id_upper', right_on=clinical_patient_col, how='inner')

    # Filtraggio tessuti se richiesto 'single_tissue'
    if tissue_mode == 'single_tissue':
        selected_rows = []
        for p_id, group in merged_df.groupby('patient_id_upper'):
            selected_rows.append(select_most_fitting_sample(group))
        merged_df = pd.DataFrame(selected_rows).reset_index(drop=True)
        logging.info(
            f"[{tissue_mode.upper()}] Selezionato 1 solo tessuto per paziente -> {len(merged_df)} campioni totali.")
    else:
        logging.info(
            f"[{tissue_mode.upper()}] Utilizzati tutti i tessuti disponibili -> {len(merged_df)} campioni totali.")

    # Caricamento grafi .pt da disco in RAM
    samples_data = []
    labels = []
    sample_ids = []
    patient_ids = []

    label_enc = LabelEncoder()
    merged_df['target_encoded'] = label_enc.fit_transform(merged_df[target_col])
    logging.info(
        f"Target '{target_col}' Mappatura Classi: {dict(zip(label_enc.classes_, range(len(label_enc.classes_))))}")

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Caricamento Grafi in RAM"):
        s_id = row['sample_id']
        f_path = os.path.join(graphs_dir, f"{s_id}.pt")

        if os.path.exists(f_path):
            graph = torch.load(f_path, weights_only=False)
            # graph.x ha forma [Num_Genes, 2] dove col 0 = RNA, col 1 = CNV
            samples_data.append(graph.x.numpy())
            labels.append(row['target_encoded'])
            sample_ids.append(s_id)
            patient_ids.append(row['patient_id_upper'])

    return np.array(samples_data), np.array(labels), sample_ids, patient_ids, len(label_enc.classes_)


# ==============================================================================
# 4. TRAIN & EVALUATION PIPELINE
# ==============================================================================

def run_single_experiment(raw_x_data, y_labels, patient_ids, modality, target_feat, num_classes, n_splits=5):
    """
    Esegue una 5-Fold Cross Validation per una specifica modalità omica con selezione delle feature e checkpointing.
    """
    if modality == 'RNA_only':
        x_flat = raw_x_data[:, :, 0]
    elif modality == 'CNV_only':
        x_flat = raw_x_data[:, :, 1]
    elif modality == 'RNA_plus_CNV':
        x_flat = raw_x_data.reshape(raw_x_data.shape[0], -1)
    else:
        raise ValueError(f"Modalità non supportata: {modality}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(x_flat, y_labels, groups=patient_ids)):
        X_train, y_train = x_flat[train_idx], y_labels[train_idx]
        X_val, y_val = x_flat[val_idx], y_labels[val_idx]

        # 1. Normalizzazione (eseguita solo sul set di Train per evitare data leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 2. SELEZIONE DELLE FEATURE (Indispensabile per evitare overfitting con N << P)
        # Seleziona i 100 geni/feature più informativi sul Train
        selector = SelectKBest(score_func=f_classif, k=100)
        X_train = selector.fit_transform(X_train, y_train)
        X_val = selector.transform(X_val)

        # Datasets & Loaders
        train_ds = InMemoryOmicDataset(X_train, y_train, None, None)
        val_ds = InMemoryOmicDataset(X_val, y_val, None, None)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # Inizializzazione Modello con input_dim ridotta a 100
        model = TabularGeneMLP(input_dim=X_train.shape[1], hidden_dim=64, num_classes=num_classes, dropout=0.5).to(
            device)

        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = torch.tensor(len(y_train) / (num_classes * np.maximum(class_counts, 1)),
                                     dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-1)

        # Loop di addestramento con Early Stopping e Model Checkpointing
        best_val_loss = float('inf')
        best_metrics = {}
        best_model_weights = None
        patience = 10
        patience_counter = 0

        history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

        for epoch in range(1, 101):
            model.train()
            running_train_loss = 0.0
            train_preds = []
            train_targets = []

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * bx.size(0)
                preds = torch.argmax(out, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(by.cpu().numpy())

            epoch_train_loss = running_train_loss / len(train_ds)
            epoch_train_acc = accuracy_score(train_targets, train_preds)

            # Valutazione su Validation Set
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss += loss.item() * bx.size(0)

                    probs = torch.softmax(out, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy() if num_classes == 2 else probs.cpu().numpy())
                    all_targets.extend(by.cpu().numpy())

            val_loss /= len(val_ds)
            acc = accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

            try:
                auc = roc_auc_score(all_targets, all_probs) if num_classes == 2 else np.nan
            except ValueError:
                auc = np.nan

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['train_metric'].append(epoch_train_acc)
            history['val_metric'].append(acc)

            # CHECKPOINTING: Salva una COPIA PROFONDA dei pesi migliori
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {'acc': acc, 'f1': f1, 'auc': auc, 'loss': val_loss}
                best_model_weights = copy.deepcopy(model.state_dict())  # <--- Salva snapshot
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # RIPRISTINO PESI MIGLIORI: da inserire FUORI dal loop delle epoche
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)  # <--- Ripristina i pesi a min val_loss

        save_learning_curves(history, modality, target_feat, fold + 1, task_name='classification')
        fold_metrics.append(best_metrics)

    # Aggregazione metriche tra i fold
    mean_acc = np.mean([m['acc'] for m in fold_metrics])
    std_acc = np.std([m['acc'] for m in fold_metrics])
    mean_f1 = np.mean([m['f1'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])
    mean_auc = np.nanmean([m['auc'] for m in fold_metrics])
    std_auc = np.nanstd([m['auc'] for m in fold_metrics])

    return {
        'Accuracy': f"{mean_acc:.4f} ± {std_acc:.4f}",
        'F1-Macro': f"{mean_f1:.4f} ± {std_f1:.4f}",
        'ROC-AUC': f"{mean_auc:.4f} ± {std_auc:.4f}"
    }


# ==============================================================================
# 5. ESECUZIONE BENCHMARK COMPLETO
# ==============================================================================

def main():
    graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR
    preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR

    metadata_path = os.path.join(graphs_dir, "associated_metadata.tsv")
    clinical_path = os.path.join(preprocessed_dir, "preprocessed_clinical_data.tsv")

    if not os.path.exists(metadata_path) or not os.path.exists(clinical_path):
        raise FileNotFoundError("Assicurati che associated_metadata.tsv e preprocessed_clinical_data.tsv esistano.")

    metadata_df = pd.read_csv(metadata_path, sep='\t')
    clinical_df = pd.read_csv(clinical_path, sep='\t')

    targets_to_test = ['is_platinum_resistant', 'Primary_therapy_outcome']
    tissue_modes = ['single_tissue', 'all_tissues']
    modalities = ['RNA_only', 'CNV_only', 'RNA_plus_CNV']

    results = []

    for target in targets_to_test:
        if target not in clinical_df.columns:
            logging.warning(f"Target '{target}' non presente nel file clinico. Salto.")
            continue

        for tissue_mode in tissue_modes:
            logging.info(f"=======================================================")
            logging.info(f" TARGET: {target} | TISSUE MODE: {tissue_mode}")
            logging.info(f"=======================================================")

            raw_x, y_labels, s_ids, p_ids, num_classes = load_raw_omic_data(
                graphs_dir, metadata_df, clinical_df, target, tissue_mode
            )

            for mod in modalities:
                logging.info(f"--> Valutazione MLP su Modalità: {mod}...")
                metrics = run_single_experiment(raw_x, y_labels, np.array(p_ids), mod, target, num_classes)

                results.append({
                    'Target': target,
                    'Tissue_Mode': tissue_mode,
                    'Modality': mod,
                    'Accuracy': metrics['Accuracy'],
                    'F1-Macro': metrics['F1-Macro'],
                    'ROC-AUC': metrics['ROC-AUC']
                })

    # Stampa e salvataggio Report Finale
    df_results = pd.DataFrame(results)
    print("\n\n=======================================================")
    print("           MLP BENCHMARK COMPARISON REPORT            ")
    print("=======================================================")
    print(df_results.to_string(index=False))

    out_csv = "mlp_benchmark_results.csv"
    df_results.to_csv(out_csv, index=False)
    logging.info(f"Report risultati salvato con successo in: {out_csv}")

    # GENERAZIONE E SALVATAGGIO IMMAGINI PNG (TABELLE E HEATMAP)
    save_benchmark_visualizations(df_results, output_dir='results_visualizations')


if __name__ == "__main__":
    main()