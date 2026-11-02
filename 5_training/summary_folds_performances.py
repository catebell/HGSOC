import os
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAZIONE INPUT
# Inserire qui i modelli e i relativi percorsi dei file CSV di report k-fold
# ==============================================================================
MODELS_CSV = {
    "GAT tissue_type": "trained_models/22_10_2026_GAT_tissue_type/GAT/report_kfold_test_results_classification.csv",
    "MIL tissue_type": "trained_models/22_10_2026_testMIL_fromGAT/MIL/report_kfold_test_results_MIL.csv",
    "MLP tissue_type": "/home/ov24456/PycharmProjects/project/test_ABMIL/trained_models/20_10_2026/SampleMLP/report_kfold_test_results_classification.csv",
    "GAT is_plat_res": "/home/ov24456/PycharmProjects/project/5_training/trained_models/22_10_2026_GAT_is_platinum_res/GAT/report_kfold_test_results_classification.csv",
    "GAT is_plat_res NACT": "trained_models/22_10_2026_GAT_platRes_onlyNACT/GAT/report_kfold_test_results_GAT.csv",
    "GAT is_plat_res PDS": "trained_models/22_10_2026_GAT_platRes_onlyPDS/GAT/report_kfold_test_results_GAT.csv"
}

# File di output
OUTPUT_CSV_PATH = "aggregated_metrics_summary_GAT_platRes.csv"
OUTPUT_PNG_PATH = "aggregated_metrics_summary_GAT_platRes.png"

# ==============================================================================
# FUNZIONE DI ELABORAZIONE E AGGREGAZIONE
# ==============================================================================
def aggregate_metrics(models_dict):
    records = []

    # Mappatura dei possibili nomi delle colonne nei file CSV di input
    col_mapping = {
        'test_loss': 'Loss',
        'loss': 'Loss',
        'acc': 'Accuracy',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'f1_score': 'F1 Score',
        'auc': 'ROC AUC',
        'roc_auc': 'ROC AUC'
    }

    target_columns = ['Accuracy', 'F1 Score', 'ROC AUC', 'Loss']

    for model_name, csv_path in models_dict.items():
        if not os.path.exists(csv_path):
            print(f"[WARNING] File non trovato: '{csv_path}' per il modello '{model_name}'. Saltato.")
            continue

        # Lettura file CSV
        df = pd.read_csv(csv_path)

        # Normalizzazione nomi colonne
        df_renamed = df.rename(columns=lambda c: col_mapping.get(c.lower().strip(), c))

        row = {'Modello': model_name}

        # Calcolo di Media ± Deviazione Standard per ogni metrica
        for col in target_columns:
            if col in df_renamed.columns:
                mean_val = df_renamed[col].mean()
                std_val = df_renamed[col].std() # Sample standard deviation (ddof=1)
                row[col] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[col] = "N/A"

        records.append(row)

    if not records:
        raise ValueError("Nessun file CSV è stato letto con successo. Verificare i percorsi inseriti in MODELS_CSV.")

    df_summary = pd.DataFrame(records)
    return df_summary

# ==============================================================================
# SALVATAGGIO TABELLA GRAFICA PNG
# ==============================================================================
def save_table_png(df_summary, output_png_path):
    # Calcolo dinamico delle dimensioni della figura in base al numero di modelli
    fig_height = max(1.8, 1.2 + len(df_summary) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
    ax.axis('off')

    # Creazione della tabella Matplotlib
    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Stile visivo: Intestazione scura e righe alternate
    header_color = '#2B4C7E'  # Blu scuro professionale
    alt_row_color = '#F0F4F8' # Grigio/azzurro chiarissimo

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
    print(f"[OK] Immagine tabella salvata in: '{output_png_path}'")

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