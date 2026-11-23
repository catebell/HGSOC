import os
import pandas as pd
import matplotlib.pyplot as plt

import config_file_tree

# ==============================================================================
# CONFIGURAZIONE INPUT
# Inserire qui i modelli e i relativi percorsi dei file CSV di report k-fold
# ==============================================================================

MODELS_CSV = {
    "MIL PDS unweighted": "/home/ov24456/PycharmProjects/project/tests/with SMOTE/no subsampling/16_11_2026_MIL_noSubsampling_PDS_unweighted/MIL/report_kfold_test_results_MIL.csv",
    "MIL PDS weights": "/home/ov24456/PycharmProjects/project/tests/with SMOTE/no subsampling/19_11_2026_MIL_PDS_class_weights/MIL/report_kfold_test_results_MIL.csv",
    "MIL PDS smote": "/home/ov24456/PycharmProjects/project/tests/with SMOTE/no subsampling/19_11_2026_MIL_PDS_latent_smote/MIL/report_kfold_test_results_MIL.csv",

    "MIL NACT unweighted": "/home/ov24456/PycharmProjects/project/tests/with SMOTE/no subsampling/19_11_2026_MIL_NACT_unweighted/MIL/report_kfold_test_results_MIL.csv",
    "MIL NACT weights": "/home/ov24456/PycharmProjects/project/tests/with SMOTE/no subsampling/19_11_2026_MIL_NACT_class_weights/MIL/report_kfold_test_results_MIL.csv",
    "MIL NACT smote": "/home/ov24456/PycharmProjects/project/tests/with SMOTE/no subsampling/19_11_2026_MIL_NACT_latent_smote/MIL/report_kfold_test_results_MIL.csv",
}

# File di output
OUTPUT_CSV_PATH = os.path.join(config_file_tree.EXPLAINABILITY_DIR, "aggregated_metrics_comparison.csv")
OUTPUT_PNG_PATH = os.path.join(config_file_tree.EXPLAINABILITY_DIR, "aggregated_metrics_comparison.png")

def aggregate_metrics(models_dict):
    records = []

    # Mapping of input CSV column names
    col_mapping = {
        'test_loss': 'Loss',
        'loss': 'Loss',
        'acc': 'Accuracy',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'f1_score': 'F1 Score',
        'auc': 'ROC AUC',
        'roc_auc': 'ROC AUC',
        'c_index': 'C-Index',
        'cindex': 'C-Index'
    }

    target_columns = ['Accuracy', 'F1 Score', 'ROC AUC', 'C-Index', 'Loss']

    for model_name, csv_path in models_dict.items():
        if not os.path.exists(csv_path):
            print(f"[WARNING] File non trovato: '{csv_path}' per il modello '{model_name}'. Saltato.")
            continue

        df = pd.read_csv(csv_path)

        # Filter out legacy summary rows if present
        if 'fold' in df.columns:
            fold_col_str = df['fold'].astype(str).str.strip().str.lower()
            summary_labels = {'mean', 'std', 'var', 'variance', 'mean ± var', 'mean +- var'}
            df = df[~fold_col_str.isin(summary_labels)].copy()

        df_renamed = df.rename(columns=lambda c: col_mapping.get(c.lower().strip(), c))

        row = {'Modello': model_name}

        # Calculate Mean ± Std for present metrics
        for col in target_columns:
            if col in df_renamed.columns:
                vals = pd.to_numeric(df_renamed[col], errors='coerce').dropna()
                if not vals.empty:
                    mean_val = vals.mean()
                    std_val = vals.std() if len(vals) > 1 else 0.0
                    row[col] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[col] = "N/A"

        # Exclude columns not present in the current experiment set
        records.append(row)

    if not records:
        raise ValueError("Nessun file CSV è stato letto con successo. Verificare i percorsi inseriti.")

    df_summary = pd.DataFrame(records)
    # Drop columns that are completely N/A across all models
    df_summary = df_summary.dropna(how='all', axis=1)
    df_summary = df_summary.loc[:, (df_summary != "N/A").any(axis=0)]
    return df_summary


def save_table_png(df_summary, output_png_path):
    fig_height = max(1.8, 1.2 + len(df_summary) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
    ax.axis('off')

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    header_color = '#2B4C7E'
    alt_row_color = '#F0F4F8'

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row_idx % 2 == 0:
                cell.set_facecolor(alt_row_color)
            else:
                cell.set_facecolor('#FFFFFF')

    plt.title("Aggregated K-Fold Performance Metrics (Mean ± Std)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_png_path, bbox_inches='tight', dpi=300)
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Elaborazione metriche in corso...")
    df_summary = aggregate_metrics(MODELS_CSV)

    # 1. Salvataggio su file CSV
    df_summary.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[OK] CSV di output salvato in: '{OUTPUT_CSV_PATH}'")

    # 2. Salvataggio tabella grafica PNG
    save_table_png(df_summary, OUTPUT_PNG_PATH)

    print("\n--- TABELLA RIASSUNTIVA ---")
    print(df_summary.to_string(index=False))