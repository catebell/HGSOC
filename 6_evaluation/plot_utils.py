import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config_file_tree

logging.getLogger('matplotlib').setLevel(logging.WARNING)

def plot_importance_barplot(df_importance, x_col, y_col, title, output_path, top_n=20, orientation='horizontal'):
    """
    Generate a clean barplot for top-N feature/gene importances or saliency scores.
    Adds a subtle black reference line at 0 to distinguish positive/negative contributions.
    """
    if df_importance.empty or 'not used' in df_importance[x_col].values:
        logging.info(f"Skipping plot for {title}: data is empty or not used.")
        return

    df_top = df_importance.head(top_n).copy()

    # Assicura il tipo numerico esplicito
    df_top[x_col] = pd.to_numeric(df_top[x_col], errors='coerce')

    plt.figure(figsize=(10, 6))

    if orientation == 'vertical':
        sns.barplot(
            data=df_top,
            x=y_col,
            y=x_col,
            palette='RdPu_r',
            hue=y_col,
            legend=False
        )
        # Linea sottile nera di riferimento sullo 0 per i barplot verticali
        plt.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.8, zorder=3)

        plt.xticks(rotation=45, ha='right')
        plt.xlabel(y_col.replace('_', ' ').capitalize(), fontsize=12)
        plt.ylabel(x_col.replace('_', ' ').capitalize(), fontsize=12)
    else:
        sns.barplot(
            data=df_top,
            x=x_col,
            y=y_col,
            palette='RdPu_r',
            hue=y_col,
            legend=False
        )
        # Linea sottile nera di riferimento sullo 0 per i barplot orizzontali
        plt.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.8, zorder=3)

        plt.xlabel(x_col.replace('_', ' ').capitalize(), fontsize=12)
        plt.ylabel(y_col.replace('_', ' ').capitalize(), fontsize=12)

    plt.title(f"{title} (Top {min(top_n, len(df_top))})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved barplot to {output_path}")


def collect_top_genes_node_features(loader, top_gene_ids, ensg_to_idx, num_unique_genes, is_mil=False):
    """
    Extract node feature values (e.g. gene expression, CNV) across samples/patients
    for the specified top gene IDs.

    If is_mil=True, averages gene feature values across all tissue samples belonging to the same patient.
    """
    records = []

    if not is_mil:
        # Sample-level DataLoader
        for batch in loader:
            batch_y = batch.y.cpu().numpy()
            batch_x = batch.x.cpu().numpy()  # Shape: [num_graphs * num_unique_genes, num_features]
            num_graphs = batch.num_graphs

            for g_idx in range(num_graphs):
                label = int(batch_y[g_idx])
                graph_x = batch_x[g_idx * num_unique_genes: (g_idx + 1) * num_unique_genes]

                row = {'label': label}
                for gene_id in top_gene_ids:
                    if gene_id in ensg_to_idx:
                        node_idx = ensg_to_idx[gene_id]
                        row[gene_id] = graph_x[node_idx]  # Array of feature values for this gene
                    else:
                        row[gene_id] = np.nan
                records.append(row)
    else:
        # Patient-level MIL DataLoader
        for patient_batch, target_y, patient_id in loader:
            label = int(target_y.item())
            patient_x = patient_batch.x.cpu().numpy()  # Shape: [total_tissue_nodes, num_features]
            num_tissue_samples = patient_batch.batch.max().item() + 1

            # Reshape patient tissues: [num_tissues, num_unique_genes, num_features]
            tissues_x = patient_x.reshape(num_tissue_samples, num_unique_genes, -1)
            # Average gene feature values across all tissues of the patient
            patient_mean_x = tissues_x.mean(axis=0)  # Shape: [num_unique_genes, num_features]

            row = {'label': label}
            for gene_id in top_gene_ids:
                if gene_id in ensg_to_idx:
                    node_idx = ensg_to_idx[gene_id]
                    row[gene_id] = patient_mean_x[node_idx]
                else:
                    row[gene_id] = np.nan
            records.append(row)

    return records


def plot_gene_feature_boxplots(records, top_genes_dict, class_names, output_dir, prefix="gat", feature_names=None):
    """
    Generates boxplots + stripplots comparing feature values across target classes for each top gene.

    top_genes_dict: dict mapping gene_id -> gene_name/alias
    feature_names: list of feature names corresponding to indices in data.x (e.g. ['RNA Expression', 'CNV']).
                   If None, automatically retrieves config_file_tree.NODE_FEATURE_NAMES.
    """
    if not records:
        logging.warning("No feature records collected for boxplots.")
        return

    # Check number of node features present in data.x
    sample_gene_val = list(records[0].values())[1]  # Get first gene feature value
    num_node_features = len(sample_gene_val) if isinstance(sample_gene_val, (np.ndarray, list)) else 1

    # Automatic fallback to config_file_tree.NODE_FEATURE_NAMES
    if feature_names is None and hasattr(config_file_tree, 'NODE_FEATURE_NAMES'):
        feature_names = config_file_tree.NODE_FEATURE_NAMES

    if feature_names is None or len(feature_names) != num_node_features:
        feature_names = [f"Feature_{i}" for i in range(num_node_features)]

    # Generate boxplots for each feature index in node features
    for f_idx, feat_name in enumerate(feature_names):
        # Build DataFrame for this feature index
        df_rows = []
        for r in records:
            row_dict = {'label': r['label']}
            for gene_id in top_genes_dict.keys():
                val = r[gene_id]
                row_dict[gene_id] = val[f_idx] if isinstance(val, (np.ndarray, list)) else val
            df_rows.append(row_dict)

        df_plot = pd.DataFrame(df_rows)

        # Conversione numerica esplicita per evitare warning di Matplotlib su unità categoriali
        for gene_id in top_genes_dict.keys():
            df_plot[gene_id] = pd.to_numeric(df_plot[gene_id], errors='coerce').astype(float)
        df_plot['label'] = pd.to_numeric(df_plot['label'], errors='coerce').astype(int)

        # Plot setup
        num_genes = len(top_genes_dict)
        plt.figure(figsize=(4 * num_genes, 6))
        plt.rc('font', size=11)

        unique_labels = sorted(df_plot['label'].unique())
        class_labels_map = {lbl: (class_names[lbl] if lbl < len(class_names) else str(lbl)) for lbl in unique_labels}

        for j, (gene_id, gene_symbol) in enumerate(top_genes_dict.items(), 1):
            plt.subplot(1, num_genes, j)

            sns.boxplot(
                x='label',
                y=gene_id,
                data=df_plot,
                palette='Spectral',
                hue='label',
                legend=False
            )
            sns.stripplot(
                x='label',
                y=gene_id,
                data=df_plot,
                color='black',
                size=3,
                alpha=0.5,
                jitter=0.2
            )

            plt.title(f"{gene_symbol}\n({gene_id})", fontsize=12, fontweight='bold')
            plt.xticks(
                ticks=range(len(unique_labels)),
                labels=[class_labels_map[lbl] for lbl in unique_labels],
                rotation=15
            )
            plt.xlabel("Target Class", fontsize=11)
            plt.ylabel(str(feat_name), fontsize=11)

        plt.tight_layout()
        feat_filename = str(feat_name).lower().replace(' ', '_').replace('/', '_')
        out_filename = os.path.join(output_dir, f"{prefix}_top5_genes_{feat_filename}_boxplot.png")
        plt.savefig(out_filename, dpi=300)
        plt.close()
        logging.info(f"Saved top 5 genes boxplot to {out_filename}")