import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import to_hex, to_rgb
from pyvis.network import Network
from tqdm import tqdm
import torch

import config_file_tree

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# =============================================================================
# GLOBAL FILE NAME CONSTANTS FOR EXPLAINABILITY OUTPUTS
# =============================================================================
FN_GAT_EDGE_ATTENTION = "gat_edge_attention.tsv"
FN_MIL_EDGE_ATTENTION = "mil_edge_attention.tsv"

FN_GAT_GENE_ATTENTION_MAP = "gat_gene_feature_attention_map.tsv"
FN_MIL_GENE_ATTENTION_MAP = "mil_gene_feature_attention_map.tsv"

FN_GAT_METADATA_IMPORTANCE_TSV = "gat_sample_metadata_importance.tsv"
FN_GAT_METADATA_IMPORTANCE_PNG = "gat_sample_metadata_importance.png"

FN_MIL_CLINICAL_IMPORTANCE_TSV = "mil_clinical_features_importance.tsv"
FN_MIL_CLINICAL_IMPORTANCE_PNG = "mil_clinical_features_importance.png"

FN_GAT_ALL_SHAP_TSV = "gat_all_features_shap.tsv"
FN_GAT_ALL_SHAP_PNG = "gat_all_features_shap_summary.png"
FN_MIL_ALL_SHAP_TSV = "mil_all_features_shap.tsv"
FN_MIL_ALL_SHAP_PNG = "mil_all_features_shap_summary.png"

FN_GAT_NET_INTERACTIVE_HTML = "gat_gene_network_interactive.html"
FN_GAT_NET_STATIC_PNG = "gat_gene_network_static.png"
FN_MIL_NET_INTERACTIVE_HTML = "mil_gene_network_interactive.html"
FN_MIL_NET_STATIC_PNG = "mil_gene_network_static.png"

FN_GAT_HEATMAP_PNG = "gat_gene_feature_attention_heatmap.png"
FN_MIL_HEATMAP_PNG = "mil_gene_feature_attention_heatmap.png"

FN_KEGG_BARPLOT_PNG = "kegg_pathways_barplot.png"
FN_KEGG_TOP_PATHWAYS_TSV = "kegg_top_pathways.tsv"

FN_USER_GENE_SUBSETS_JSON = "user_gene_subsets.json"

FN_MIL_TISSUE_ATTENTION_AGGREGATION = "mil_tissue_attention_aggregation.tsv"
FN_MIL_TISSUE_ATTENTION_BARPLOT_PNG = "mil_tissue_attention_distribution.png"

# =============================================================================
# DYNAMIC FILENAME RESOLVER HELPERS
# =============================================================================
def get_performance_report_filenames(prefix):
    """Return a list of potential evaluation report file names for a given model prefix."""
    prefix_str = str(prefix).lower()
    return [
        f"report_kfold_test_results_{prefix_str.upper()}.csv",
        "report_kfold_test_results.csv",
        f"report_kfold_test_results_{config_file_tree.TASK}.csv"
    ]


def get_gene_shap_tsv_filename(prefix, feat_name):
    """Construct gene SHAP TSV filename dynamically for a given feature."""
    clean_feat_name = str(feat_name).replace(' ', '_').lower()
    return f"{str(prefix).lower()}_gene_{clean_feat_name}_shap.tsv"


def get_gene_shap_png_filename(prefix, feat_name):
    """Construct gene SHAP beeswarm plot filename dynamically for a given feature."""
    clean_feat_name = str(feat_name).replace(' ', '_').lower()
    return f"{str(prefix).lower()}_gene_{clean_feat_name}_shap_summary.png"


def get_top5_boxplot_png_filename(prefix, feat_name):
    """Construct top 5 genes boxplot image filename dynamically for a given feature."""
    clean_feat_name = str(feat_name).lower().replace(' ', '_').replace('/', '_')
    return f"{str(prefix).lower()}_top5_genes_{clean_feat_name}_boxplot.png"


def get_importance_tsv_filename(prefix):
    """Return the tabular feature importance TSV filename based on model prefix."""
    return FN_GAT_METADATA_IMPORTANCE_TSV if str(prefix).lower() == "gat" else FN_MIL_CLINICAL_IMPORTANCE_TSV


def get_importance_png_filename(prefix):
    """Return the feature importance barplot image filename based on model prefix."""
    return FN_GAT_METADATA_IMPORTANCE_PNG if str(prefix).lower() == "gat" else FN_MIL_CLINICAL_IMPORTANCE_PNG


def get_edge_attention_tsv_filename(prefix):
    """Return PPI edge attention TSV filename for GAT or MIL."""
    return FN_GAT_EDGE_ATTENTION if str(prefix).lower() == "gat" else FN_MIL_EDGE_ATTENTION


def get_gene_attention_map_tsv_filename(prefix):
    """Return Gene x Feature attention map TSV filename for GAT or MIL."""
    return FN_GAT_GENE_ATTENTION_MAP if str(prefix).lower() == "gat" else FN_MIL_GENE_ATTENTION_MAP


def get_all_shap_tsv_filename(prefix):
    """Return unified all-features SHAP TSV filename for GAT or MIL."""
    return FN_GAT_ALL_SHAP_TSV if str(prefix).lower() == "gat" else FN_MIL_ALL_SHAP_TSV


def get_all_shap_png_filename(prefix):
    """Return unified all-features SHAP beeswarm plot filename for GAT or MIL."""
    return FN_GAT_ALL_SHAP_PNG if str(prefix).lower() == "gat" else FN_MIL_ALL_SHAP_PNG


def get_heatmap_png_filename(prefix):
    """Return 2D Gene x Feature attention heatmap filename for GAT or MIL."""
    return FN_GAT_HEATMAP_PNG if str(prefix).lower() == "gat" else FN_MIL_HEATMAP_PNG


def get_net_interactive_html_filename(prefix):
    """Return interactive network HTML filename for GAT or MIL."""
    return FN_GAT_NET_INTERACTIVE_HTML if str(prefix).lower() == "gat" else FN_MIL_NET_INTERACTIVE_HTML


def get_net_static_png_filename(prefix):
    """Return static network PNG filename for GAT or MIL."""
    return FN_GAT_NET_STATIC_PNG if str(prefix).lower() == "gat" else FN_MIL_NET_STATIC_PNG


def get_mil_tissue_attention_aggregation_tsv_filename():
    """Return TSV filename for MIL tissue-level attention aggregation."""
    return FN_MIL_TISSUE_ATTENTION_AGGREGATION


# =============================================================================
# EXPLAINABILITY PIPELINE FUNCTIONS
# =============================================================================
def to_rel_path(path_str, root_dir=None):
    """Convert an absolute file path to a path relative to the project root directory."""
    if not path_str:
        return ""
    if root_dir is None:
        root_dir = config_file_tree.ROOT_DIR
    try:
        return os.path.relpath(path_str, root_dir)
    except Exception:
        return str(path_str)


def load_gene_mappings(gene_mapping_file):
    """Load gene node mapping dictionary from TSV."""
    gene_map_dict = {}
    if os.path.exists(gene_mapping_file):
        df_map = pd.read_csv(gene_mapping_file, sep='\t')
        gene_map_dict = {
            int(row['node_index']): (str(row['gene_id']), str(row['gene_symbol']))
            for _, row in df_map.iterrows()
        }
    return gene_map_dict


def get_gene_name(gene_idx, gene_map_dict):
    """Retrieve gene ID and symbol for a given node index."""
    if gene_idx in gene_map_dict:
        return gene_map_dict[gene_idx]
    return f"GENE_{gene_idx}", f"Unknown_{gene_idx}"


def extract_gat_edge_attention(model, loader, gene_map_dict, num_unique_genes, output_dir, device):
    """Extract GAT PPI edge attention weights and save to TSV."""
    logging.info("Extracting GAT PPI edge attention weights...")
    model.eval()
    edge_attention_sum, edge_counts = {}, {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="GAT Edge Attention"):
            batch = batch.to(device)
            node_indices = torch.arange(batch.x.size(0), device=device) % num_unique_genes
            gene_emb = model.gene_embedding(node_indices)
            x_input = torch.cat([batch.x, gene_emb], dim=-1)

            out, (edge_index, alpha) = model.conv1(x_input, batch.edge_index, return_attention_weights=True)
            mean_alpha = alpha.mean(dim=1).cpu().numpy()

            src_nodes = (edge_index[0] % num_unique_genes).cpu().numpy()
            dst_nodes = (edge_index[1] % num_unique_genes).cpu().numpy()

            for i in range(len(mean_alpha)):
                u, v = int(src_nodes[i]), int(dst_nodes[i])
                if u == v:
                    continue
                edge_key = tuple(sorted((u, v)))
                edge_attention_sum[edge_key] = edge_attention_sum.get(edge_key, 0.0) + float(mean_alpha[i])
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

    results = []
    for (u, v), total_att in edge_attention_sum.items():
        avg_att = total_att / edge_counts[(u, v)]
        gene_id_a, symbol_a = get_gene_name(u, gene_map_dict)
        gene_id_b, symbol_b = get_gene_name(v, gene_map_dict)
        results.append({
            'gene_id_a': gene_id_a, 'gene_symbol_a': symbol_a,
            'gene_id_b': gene_id_b, 'gene_symbol_b': symbol_b,
            'interaction': f"{symbol_a} - {symbol_b}",
            'attention_weight': float(avg_att)
        })

    df_edges = pd.DataFrame(results).sort_values(by='attention_weight', ascending=False)
    out_path = os.path.join(output_dir, FN_GAT_EDGE_ATTENTION)
    df_edges.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved GAT edge attention to {to_rel_path(out_path)}")
    return df_edges


def extract_mil_edge_attention(model, loader, gene_map_dict, num_unique_genes, output_dir, device):
    """Extract MIL patient-level edge attention weights and save to TSV."""
    logging.info("Extracting MIL patient-level edge attention weights...")
    model.eval()
    edge_attention_sum = {}
    total_bags = 0

    with torch.no_grad():
        for item in tqdm(loader, desc="MIL Edge Attention"):
            patient_batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
            logits, attn_weights = model(patient_batch)
            attn_weights = attn_weights.squeeze()

            sample_encoder = model.sample_encoder
            if hasattr(sample_encoder, 'gene_embedding'):
                node_indices = torch.arange(patient_batch.x.size(0), device=device) % num_unique_genes
                gene_emb = sample_encoder.gene_embedding(node_indices)
                x_input = torch.cat([patient_batch.x, gene_emb], dim=-1)
            else:
                x_input = patient_batch.x

            _, (edge_index, alpha) = sample_encoder.conv1(x_input, patient_batch.edge_index,
                                                          return_attention_weights=True)
            mean_alpha = alpha.mean(dim=1)

            sample_batch_ids = patient_batch.batch[edge_index[1]]
            sample_weights = attn_weights[sample_batch_ids] if attn_weights.ndim > 0 else attn_weights
            combined_alpha = (mean_alpha * sample_weights).cpu().numpy()

            src_nodes = (edge_index[0] % num_unique_genes).cpu().numpy()
            dst_nodes = (edge_index[1] % num_unique_genes).cpu().numpy()

            for i in range(len(combined_alpha)):
                u, v = int(src_nodes[i]), int(dst_nodes[i])
                if u == v:
                    continue
                edge_key = tuple(sorted((u, v)))
                edge_attention_sum[edge_key] = edge_attention_sum.get(edge_key, 0.0) + float(combined_alpha[i])

            total_bags += 1

    results = []
    for (u, v), total_att in edge_attention_sum.items():
        avg_att = total_att / (total_bags + 1e-6)
        gene_id_a, symbol_a = get_gene_name(u, gene_map_dict)
        gene_id_b, symbol_b = get_gene_name(v, gene_map_dict)
        results.append({
            'gene_id_a': gene_id_a, 'gene_symbol_a': symbol_a,
            'gene_id_b': gene_id_b, 'gene_symbol_b': symbol_b,
            'interaction': f"{symbol_a} - {symbol_b}",
            'attention_weight': float(avg_att)
        })

    df_edges = pd.DataFrame(results).sort_values(by='attention_weight', ascending=False)
    out_path = os.path.join(output_dir, FN_MIL_EDGE_ATTENTION)
    df_edges.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved MIL edge attention to {to_rel_path(out_path)}")
    return df_edges


def extract_gene_saliency_and_attention_map(model, loader, gene_map_dict, num_unique_genes, feature_names, output_dir,
                                            device, prefix="gat"):
    """Extract 2D Gene x Feature Attention Map for GAT or MIL and save to TSV."""
    logging.info(f"Extracting {prefix.upper()} 2D Gene x Feature Attention Map...")
    model.eval()
    num_features = len(feature_names)
    feature_saliency_sum = torch.zeros((num_unique_genes, num_features), device=device)
    gene_counts = torch.zeros(num_unique_genes, device=device)

    for item in tqdm(loader, desc=f"{prefix.upper()} Gene-Feature Attention Map"):
        batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
        batch.x.requires_grad = True

        if prefix == "gat":
            metadata_x = getattr(batch, 'metadata_x', None)
            logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_x)
        else:
            logits, _ = model(batch)

        preds = logits.argmax(dim=1)
        target_scores = logits[range(len(preds)), preds].sum()

        model.zero_grad()
        target_scores.backward()

        grad_x_input = (batch.x.grad * batch.x).abs()
        node_indices = torch.arange(batch.x.size(0), device=device) % num_unique_genes

        for f_idx in range(num_features):
            feature_saliency_sum[:, f_idx].scatter_add_(0, node_indices, grad_x_input[:, f_idx])
        gene_counts.scatter_add_(0, node_indices, torch.ones(batch.x.size(0), device=device))

    avg_feature_saliency = (feature_saliency_sum / (gene_counts.unsqueeze(1) + 1e-6)).detach().cpu().numpy()
    max_val = avg_feature_saliency.max() if avg_feature_saliency.max() > 0 else 1.0
    norm_saliency = avg_feature_saliency / max_val

    results = []
    for idx in range(num_unique_genes):
        if gene_counts[idx] > 0:
            ensg_id, symbol = get_gene_name(idx, gene_map_dict)
            row_dict = {'gene_id': ensg_id, 'gene_name': symbol, 'saliency_score': float(norm_saliency[idx].mean())}
            for f_idx, feat_name in enumerate(feature_names):
                row_dict[feat_name] = float(norm_saliency[idx, f_idx])
            results.append(row_dict)

    df_map = pd.DataFrame(results).sort_values(by='saliency_score', ascending=False)
    out_path = os.path.join(output_dir, get_gene_attention_map_tsv_filename(prefix))
    df_map.to_csv(out_path, sep='\t', index=False)
    logging.info(f"Saved Gene x Feature attention map to {to_rel_path(out_path)}")
    return df_map


def compute_feature_importance(model, loader, feature_cols, output_dir, device, is_mil=False, prefix="gat",
                               feature_type_label="metadata"):
    """Compute permutation feature importance for sample metadata or patient clinical features."""
    logging.info(f"Computing {feature_type_label} feature importance...")
    out_path = os.path.join(output_dir, get_importance_tsv_filename(prefix))
    if not feature_cols:
        df_empty = pd.DataFrame([{'feature': 'not used', 'importance': 'not used'}])
        df_empty.to_csv(out_path, sep='\t', index=False)
        return

    model.eval()

    def evaluate_acc(current_loader):
        correct, total = 0, 0
        with torch.no_grad():
            for item in current_loader:
                if not is_mil:
                    batch = item.to(device)
                    metadata_x = getattr(batch, 'metadata_x', None)
                    logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_x)
                    preds = logits.argmax(dim=1)
                    correct += int((preds == batch.y).sum())
                    total += batch.num_graphs
                else:
                    patient_batch, target_y = (item[0].to(device), item[1].to(device)) if isinstance(item, (list,
                                                                                                            tuple)) else (
                        item.to(device), item.patient_y.to(device))
                    logits, _ = model(patient_batch)
                    preds = logits.argmax(dim=1)
                    correct += int((preds == target_y).sum())
                    total += target_y.size(0)
        return correct / total if total > 0 else 0.0

    baseline_acc = evaluate_acc(loader)
    feature_importances = []
    for col_idx, col_name in tqdm(enumerate(feature_cols), desc=f"{feature_type_label.capitalize()} Permutation"):
        permuted_correct, total = 0, 0
        with torch.no_grad():
            for item in loader:
                if not is_mil:
                    batch = item.to(device)
                    metadata_x = getattr(batch, 'metadata_x', None)
                    if metadata_x is not None:
                        metadata_perm = metadata_x.clone()
                        perm_idx = torch.randperm(metadata_perm.size(0))
                        metadata_perm[:, col_idx] = metadata_perm[perm_idx, col_idx]
                    else:
                        metadata_perm = None
                    logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_perm)
                    preds = logits.argmax(dim=1)
                    permuted_correct += int((preds == batch.y).sum())
                    total += batch.num_graphs
                else:
                    patient_batch, target_y = (item[0].to(device), item[1].to(device)) if isinstance(item, (list,
                                                                                                            tuple)) else (
                        item.to(device), item.patient_y.to(device))
                    if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
                        clinical_perm = patient_batch.clinical_x.clone()
                        perm_idx = torch.randperm(clinical_perm.size(0))
                        clinical_perm[:, col_idx] = clinical_perm[perm_idx, col_idx]
                        patient_batch.clinical_x = clinical_perm
                    logits, _ = model(patient_batch)
                    preds = logits.argmax(dim=1)
                    permuted_correct += int((preds == target_y).sum())
                    total += target_y.size(0)

        permuted_acc = permuted_correct / total if total > 0 else 0.0
        feature_importances.append({'feature': col_name, 'importance': float(baseline_acc - permuted_acc)})

    df_imp = pd.DataFrame(feature_importances).sort_values(by='importance', ascending=False)
    df_imp.to_csv(out_path, sep='\t', index=False)


def compute_shap_values(model, loader, feature_names, extra_cols, output_dir, device, gene_map_dict=None,
                        num_unique_genes=None, is_mil=False, prefix="gat", generate_plots=True, top_n=20):
    """Compute SHAP values for all node features and extra features."""
    logging.info(f"Computing SHAP values for {prefix.upper()} Model...")
    model.eval()
    all_sample_omic_grads, all_sample_omic_vals = [], []
    all_extra_grads, all_extra_vals = [], []
    shap_by_gene, vals_by_gene = None, None

    if gene_map_dict is not None and num_unique_genes is not None:
        num_features = len(feature_names)
        shap_by_gene = [[[] for _ in range(num_features)] for _ in range(num_unique_genes)]
        vals_by_gene = [[[] for _ in range(num_features)] for _ in range(num_unique_genes)]

    for item in loader:
        if not is_mil:
            batch = item.to(device)
            batch.x.requires_grad = True
            extra_x = getattr(batch, 'metadata_x', None)
            if extra_x is not None:
                extra_x.requires_grad = True
            logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=extra_x)
            num_samples = batch.num_graphs
            batch_ids = batch.batch.cpu().numpy()
        else:
            patient_batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
            patient_batch.x.requires_grad = True
            extra_x = getattr(patient_batch, 'clinical_x', None)
            if extra_x is not None:
                extra_x.requires_grad = True
            logits, _ = model(patient_batch)
            num_samples = logits.size(0)
            batch_ids = patient_batch.batch.cpu().numpy()
            batch = patient_batch

        preds = logits.argmax(dim=1)
        target_score = logits[range(len(preds)), preds].sum()

        model.zero_grad()
        target_score.backward()

        node_attr = (batch.x.grad * batch.x).detach().cpu().numpy()
        node_vals = batch.x.detach().cpu().numpy()

        sample_omic_attr = np.zeros((num_samples, len(feature_names)))
        sample_omic_v = np.zeros((num_samples, len(feature_names)))

        for s_idx in range(num_samples):
            mask = (batch_ids == s_idx)
            if np.any(mask):
                sample_omic_attr[s_idx] = node_attr[mask].mean(axis=0)
                sample_omic_v[s_idx] = node_vals[mask].mean(axis=0)

        all_sample_omic_grads.append(sample_omic_attr)
        all_sample_omic_vals.append(sample_omic_v)

        if shap_by_gene is not None:
            num_nodes = batch.x.size(0)
            node_indices = (torch.arange(num_nodes, device=device) % num_unique_genes).cpu().numpy()
            for idx in range(num_nodes):
                g_idx = int(node_indices[idx])
                for f_idx in range(len(feature_names)):
                    shap_by_gene[g_idx][f_idx].append(float(node_attr[idx, f_idx]))
                    vals_by_gene[g_idx][f_idx].append(float(node_vals[idx, f_idx]))

        if extra_x is not None and extra_x.grad is not None:
            all_extra_grads.append((extra_x.grad * extra_x).detach().cpu().numpy())
            all_extra_vals.append(extra_x.detach().cpu().numpy())

    if all_sample_omic_grads:
        cat_omic_grads = np.concatenate(all_sample_omic_grads, axis=0)
        cat_omic_vals = np.concatenate(all_sample_omic_vals, axis=0)
        combined_grads = cat_omic_grads
        combined_vals = cat_omic_vals
        combined_names = list(feature_names)

        if all_extra_grads:
            combined_grads = np.hstack([cat_omic_grads, np.concatenate(all_extra_grads, axis=0)])
            combined_vals = np.hstack([cat_omic_vals, np.concatenate(all_extra_vals, axis=0)])
            combined_names += list(extra_cols)

        df_combined_shap = pd.DataFrame({
            'feature': combined_names,
            'shap_importance': np.abs(combined_grads).mean(axis=0)
        }).sort_values(by='shap_importance', ascending=False)
        out_combined_tsv = os.path.join(output_dir, get_all_shap_tsv_filename(prefix))
        df_combined_shap.to_csv(out_combined_tsv, sep='\t', index=False)

        if generate_plots:
            plot_shap_summary_beeswarm(
                shap_values=combined_grads, features=combined_vals, feature_names=combined_names,
                title=f"{prefix.upper()} All Features SHAP Summary",
                output_path=os.path.join(output_dir, get_all_shap_png_filename(prefix))
            )

    if shap_by_gene is not None:
        for f_idx, feat_name in enumerate(feature_names):
            gene_shap_records = []
            for g_idx in range(num_unique_genes):
                g_shap_arr = np.array(shap_by_gene[g_idx][f_idx])
                g_mean_abs_shap = float(np.abs(g_shap_arr).mean()) if len(g_shap_arr) > 0 else 0.0
                ensg_id, symbol = get_gene_name(g_idx, gene_map_dict)
                gene_shap_records.append({
                    'gene_id': ensg_id, 'gene_name': symbol, 'gene_idx': g_idx, 'shap_importance': g_mean_abs_shap
                })

            df_gene_shap = pd.DataFrame(gene_shap_records).sort_values(by='shap_importance', ascending=False)
            out_gene_tsv = os.path.join(output_dir, get_gene_shap_tsv_filename(prefix, feat_name))
            df_gene_shap.drop(columns=['gene_idx']).to_csv(out_gene_tsv, sep='\t', index=False)

            if generate_plots:
                top_df = df_gene_shap.head(top_n)
                selected_indices = top_df['gene_idx'].values
                selected_symbols = top_df['gene_name'].values.tolist()
                sel_shap_mat = np.column_stack([np.array(shap_by_gene[g_idx][f_idx]) for g_idx in selected_indices])
                sel_val_mat = np.column_stack([np.array(vals_by_gene[g_idx][f_idx]) for g_idx in selected_indices])

                plot_shap_summary_beeswarm(
                    shap_values=sel_shap_mat, features=sel_val_mat, feature_names=selected_symbols,
                    title=f"{prefix.upper()} Top {len(selected_symbols)} Genes SHAP Summary ({feat_name})",
                    output_path=os.path.join(output_dir, get_gene_shap_png_filename(prefix, feat_name))
                )


def plot_importance_barplot(df_importance, x_col, y_col, title, output_path, top_n=20, orientation='horizontal'):
    """Generate a barplot for feature importances or saliency scores."""
    if df_importance.empty or 'not used' in df_importance[x_col].values:
        logging.info(f"Skipping plot for {title}: data is empty or not used.")
        return

    df_top = df_importance.head(top_n).copy()
    df_top[x_col] = pd.to_numeric(df_top[x_col], errors='coerce')
    plt.figure(figsize=(10, 6))

    if orientation == 'vertical':
        sns.barplot(data=df_top, x=y_col, y=x_col, palette='RdPu_r', hue=y_col, legend=False)
        plt.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.8, zorder=3)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(y_col.replace('_', ' ').capitalize(), fontsize=12)
        plt.ylabel(x_col.replace('_', ' ').capitalize(), fontsize=12)
    else:
        sns.barplot(data=df_top, x=x_col, y=y_col, palette='RdPu_r', hue=y_col, legend=False)
        plt.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.8, zorder=3)
        plt.xlabel(x_col.replace('_', ' ').capitalize(), fontsize=12)
        plt.ylabel(y_col.replace('_', ' ').capitalize(), fontsize=12)

    plt.title(f"{title} (Top {min(top_n, len(df_top))})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def collect_top_genes_node_features(loader, top_gene_ids, ensg_to_idx, num_unique_genes, is_mil=False):
    """Extract node feature values across samples/patients for specified gene IDs."""
    records = []
    if not is_mil:
        for batch in loader:
            batch_y = batch.y.cpu().numpy()
            batch_x = batch.x.cpu().numpy()
            num_graphs = batch.num_graphs
            for g_idx in range(num_graphs):
                label = int(batch_y[g_idx])
                graph_x = batch_x[g_idx * num_unique_genes: (g_idx + 1) * num_unique_genes]
                row = {'label': label}
                for gene_id in top_gene_ids:
                    row[gene_id] = graph_x[ensg_to_idx[gene_id]] if gene_id in ensg_to_idx else np.nan
                records.append(row)
    else:
        for item in loader:
            patient_batch, target_y = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item, item.patient_y)
            patient_x = patient_batch.x.cpu().numpy()
            targets_np = target_y.cpu().numpy().flatten()
            num_samples_per_patient = patient_batch.num_samples.cpu().numpy().flatten() if hasattr(patient_batch,
                                                                                                   'num_samples') and patient_batch.num_samples is not None else np.array(
                [patient_batch.batch.max().item() + 1])
            curr_node_offset = 0
            for p_idx, num_tissues in enumerate(num_samples_per_patient):
                label = int(targets_np[p_idx])
                num_nodes = int(num_tissues * num_unique_genes)
                p_x = patient_x[curr_node_offset: curr_node_offset + num_nodes]
                curr_node_offset += num_nodes
                tissues_x = p_x.reshape(int(num_tissues), num_unique_genes, -1)
                patient_mean_x = tissues_x.mean(axis=0)
                row = {'label': label}
                for gene_id in top_gene_ids:
                    row[gene_id] = patient_mean_x[ensg_to_idx[gene_id]] if gene_id in ensg_to_idx else np.nan
                records.append(row)
    return records


def plot_gene_feature_boxplots(records, top_genes_dict, class_names, output_dir, prefix="gat", feature_names=None):
    """Generates boxplots comparing feature values across target classes for each top gene."""
    if not records:
        return
    sample_gene_val = list(records[0].values())[1]
    num_node_features = len(sample_gene_val) if isinstance(sample_gene_val, (np.ndarray, list)) else 1

    # Recupera i nomi reali delle feature se non forniti esplicitamente
    if feature_names is None:
        if hasattr(config_file_tree, 'NODE_FEATURE_INDEX') and hasattr(config_file_tree, 'NODE_FEATURE_NAMES'):
            feature_names = [config_file_tree.NODE_FEATURE_NAMES[i] for i in config_file_tree.NODE_FEATURE_INDEX if i < len(config_file_tree.NODE_FEATURE_NAMES)]
        elif hasattr(config_file_tree, 'NODE_FEATURE_NAMES'):
            feature_names = config_file_tree.NODE_FEATURE_NAMES

    # Fallback di sicurezza se la lunghezza non corrisponde comunque
    if feature_names is None or len(feature_names) != num_node_features:
        feature_names = [f"Feature_{i}" for i in range(num_node_features)]

    for f_idx, feat_name in enumerate(feature_names):
        df_rows = []
        for r in records:
            row_dict = {'label': r['label']}
            for gene_id in top_genes_dict.keys():
                val = r[gene_id]
                row_dict[gene_id] = val[f_idx] if isinstance(val, (np.ndarray, list)) else val
            df_rows.append(row_dict)

        df_plot = pd.DataFrame(df_rows)
        for gene_id in top_genes_dict.keys():
            df_plot[gene_id] = pd.to_numeric(df_plot[gene_id], errors='coerce').astype(float)
        df_plot['label'] = pd.to_numeric(df_plot['label'], errors='coerce').astype(int)

        num_genes = len(top_genes_dict)
        plt.figure(figsize=(4 * num_genes, 6))
        plt.rc('font', size=11)
        unique_labels = sorted(df_plot['label'].unique())
        class_labels_map = {lbl: (class_names[lbl] if lbl < len(class_names) else str(lbl)) for lbl in unique_labels}

        for j, (gene_id, gene_symbol) in enumerate(top_genes_dict.items(), 1):
            plt.subplot(1, num_genes, j)
            sns.boxplot(x='label', y=gene_id, data=df_plot, palette='Spectral', hue='label', legend=False)
            sns.stripplot(x='label', y=gene_id, data=df_plot, color='black', size=3, alpha=0.5, jitter=0.2)
            plt.title(f"{gene_symbol}\n({gene_id})", fontsize=12, fontweight='bold')
            plt.xticks(ticks=range(len(unique_labels)), labels=[class_labels_map[lbl] for lbl in unique_labels],
                       rotation=15)
            plt.xlabel("Target Class", fontsize=11)
            plt.ylabel(str(feat_name), fontsize=11)

        plt.tight_layout()
        out_filename = os.path.join(output_dir, get_top5_boxplot_png_filename(prefix, feat_name))
        plt.savefig(out_filename, dpi=300)
        plt.close()


def plot_interactive_gene_network(tsv_path, output_html, top_n=30, saliency_tsv_path=None):
    """Plot an interactive network with PyVis from edge attention TSV."""
    if not os.path.exists(tsv_path):
        return
    df = pd.read_csv(tsv_path, sep='\t').head(top_n)
    saliency_dict = {}
    if saliency_tsv_path and os.path.exists(saliency_tsv_path):
        df_sal = pd.read_csv(saliency_tsv_path, sep='\t')
        saliency_dict = dict(zip(df_sal['gene_name'], df_sal['saliency_score']))

    G = nx.Graph()
    min_w, max_w = df['attention_weight'].min(), df['attention_weight'].max()
    denom_w = (max_w - min_w) if max_w != min_w else 1.0

    for _, row in df.iterrows():
        u, v, raw_w = str(row['gene_symbol_a']), str(row['gene_symbol_b']), float(row['attention_weight'])
        norm_width = 4.0 + 4.0 * ((raw_w - min_w) / denom_w)
        G.add_edge(u, v, weight=raw_w, width=norm_width, title=f"{u} - {v}\nAttention Weight: {raw_w:.5f}",
                   color={'color': 'rgba(74, 123, 176, 0.85)', 'highlight': '#e63946'})

    c_start, c_end = np.array(to_rgb("#B3D9FF")), np.array(to_rgb("#FFB7B2"))
    degrees = dict(G.degree())

    for node in G.nodes():
        deg = degrees[node]
        raw_sal = saliency_dict.get(node, 0.5)
        color_hex = to_hex(c_start * (1.0 - raw_sal) + c_end * raw_sal)
        G.nodes[node]['size'] = 25 + 20 * raw_sal
        G.nodes[node]['color'] = {'background': color_hex, 'border': '#444444',
                                  'highlight': {'background': '#a8dda8', 'border': '#222222'}}
        G.nodes[node]['font'] = {'color': '#333333', 'size': 14, 'face': 'arial'}
        G.nodes[node]['title'] = f"Gene: {node}\nConnections: {deg}\nSaliency: {raw_sal:.5f}"

    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333", directed=False)
    net.from_nx(G)
    net.barnes_hut(gravity=-3000, central_gravity=0.2, spring_length=140)
    net.save_graph(output_html)


def plot_static_gene_network(tsv_path, output_png, top_n=30):
    """Plot a static PPI interaction network."""
    if not os.path.exists(tsv_path):
        return
    df = pd.read_csv(tsv_path, sep='\t').head(top_n)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row['gene_symbol_a']), str(row['gene_symbol_b']), weight=float(row['attention_weight']))
    pos = nx.spring_layout(G, k=0.8, seed=42)

    plt.figure(figsize=(10, 10), facecolor='white')
    nx.draw_networkx_edges(G, pos, edge_color='#888888', alpha=0.75)
    nx.draw_networkx_nodes(G, pos, node_color='#B3D9FF', node_size=700, edgecolors='#444444')
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='#222222')
    plt.title(f"Top {top_n} Gene-Gene PPI Interactions", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_gene_feature_attention_heatmap(tsv_path, output_png, feature_names, top_n=25):
    """Plot input feature attention heatmap."""
    if not os.path.exists(tsv_path):
        return
    df = pd.read_csv(tsv_path, sep='\t').head(top_n)
    df_heatmap = df.set_index('gene_name')[feature_names]

    plt.figure(figsize=(10, max(6, int(top_n * 0.35))))
    sns.heatmap(df_heatmap, cmap='YlOrRd', annot=True, fmt='.3f', linewidths=0.5)
    plt.title(f"Input Feature Attention Map (Top {top_n} Genes)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_shap_summary_beeswarm(shap_values, features, feature_names, title, output_path, max_display=20):
    """Plot SHAP summary beeswarm plot."""
    if shap_values is None or len(shap_values) == 0:
        return
    try:
        import shap
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features, feature_names=feature_names, max_display=max_display, show=False)
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.warning(f"Failed to generate SHAP beeswarm plot: {e}")
        plt.figure(figsize=(10, 6))
        plt.title(title, fontsize=14, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def run_posthoc_pathway_enrichment(saliency_tsv_path, output_dir, top_n_genes=100):
    """Executes functional enrichment (KEGG, Reactome, GO) of highest Saliency genes."""
    import gseapy as gp

    os.makedirs(output_dir, exist_ok=True)
    df_saliency = pd.read_csv(saliency_tsv_path, sep="\t")

    # Over-Representation Analysis (ORA) on Top-N Genes
    top_genes = df_saliency.sort_values(by="saliency_score", ascending=False).head(top_n_genes)["gene_name"].tolist()

    enr = gp.enrichr(
        gene_list=top_genes,
        gene_sets=['KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023'],
        organism='human',
        outdir=os.path.join(output_dir, "enrichr_results")
    )

    # Save KEGG results
    df_res = enr.results
    kegg_df = df_res[df_res['Gene_set'] == 'KEGG_2021_Human'].sort_values(by='Adjusted P-value')
    kegg_df.to_csv(os.path.join(output_dir, FN_KEGG_TOP_PATHWAYS_TSV), sep="\t", index=False)

    # Barplot of Top 10 Pathways
    if not kegg_df.empty:
        gp.barplot(
            kegg_df.head(10),
            title=f"KEGG Pathway Enrichment (Top {top_n_genes} Saliency Genes)",
            top_term=10,
            figsize=(7, 5),
            ofname=os.path.join(output_dir, FN_KEGG_BARPLOT_PNG)
        )

    # GSEA Preranked on whole gene list
    rnk = df_saliency[['gene_name', 'saliency_score']].dropna().sort_values(by='saliency_score', ascending=False)

    pre_res = gp.prerank(
        rnk=rnk,
        gene_sets=['KEGG_2021_Human'],
        processes=4,
        permutation_num=1000,
        outdir=os.path.join(output_dir, "gsea_prerank_results")
    )

    logging.info(f"[ENRICHMENT] Finished. Results saved in '{output_dir}'.")


def _unwrap_id(val):
    """Estrattore helper per convertire liste/tuple nidificate di PyG DataLoader in stringhe pulite."""
    while isinstance(val, (list, tuple, np.ndarray)):
        if len(val) > 0:
            val = val[0]
        else:
            return ""
    if hasattr(val, 'item'):
        val = val.item()
    return str(val).strip()


def extract_mil_tissue_attention_aggregation(model, loader, output_dir, device, label_encoder_sample=None):
    """
    Calculates MIL attention aggregation globally and specifically for multi-tissue mixed bags
    to prevent single-sample bag bias (where softmax forces attention = 1.0).
    """
    logging.info("Extracting MIL Attention Aggregation by Tissue Class (handling single-sample bias)...")
    model.eval()
    records = []

    with torch.no_grad():
        for item in loader:
            patient_batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)

            logits, attn_weights = model(patient_batch)
            weights = np.atleast_1d(attn_weights.squeeze().cpu().numpy())

            # Estrazione etichette tessuti (GAT embedding classes)
            if hasattr(patient_batch, 'y') and patient_batch.y is not None:
                tissue_indices = patient_batch.y.cpu().numpy()
            else:
                tissue_indices = np.array([0] * len(weights))

            tissue_names = []
            for t in tissue_indices:
                if label_encoder_sample is not None and hasattr(label_encoder_sample, 'classes_') and t < len(label_encoder_sample.classes_):
                    tissue_names.append(str(label_encoder_sample.classes_[t]))
                else:
                    tissue_names.append(f"Tissue_{t}")

            # Unwrapping sicuro degli ID paziente per evitare oggetti 'list' unhashable
            p_ids = getattr(patient_batch, 'patient_id', 'Unknown')
            if hasattr(patient_batch, 'num_samples') and patient_batch.num_samples is not None:
                num_samples_list = patient_batch.num_samples.cpu().numpy().tolist() if hasattr(patient_batch.num_samples, 'cpu') else patient_batch.num_samples
                patient_id_per_sample = []
                for p_idx, n_s in enumerate(num_samples_list):
                    if isinstance(p_ids, (list, tuple)) and p_idx < len(p_ids):
                        curr_pid = _unwrap_id(p_ids[p_idx])
                    else:
                        curr_pid = _unwrap_id(p_ids)
                    patient_id_per_sample.extend([curr_pid] * n_s)
            else:
                patient_id_per_sample = [_unwrap_id(p_ids)] * len(weights)

            for p_id, t_name, w in zip(patient_id_per_sample, tissue_names, weights):
                records.append({
                    'patient_id': p_id,
                    'tissue_type': t_name,
                    'attention_weight': float(w)
                })

    df_rec = pd.DataFrame(records)

    if not df_rec.empty:
        # Garanzia esplicita che i campi di raggruppamento siano stringhe primitive
        df_rec['patient_id'] = df_rec['patient_id'].astype(str)
        df_rec['tissue_type'] = df_rec['tissue_type'].astype(str)

        # Conteggio dimensione bag e tipi di tessuto distinti per ciascun paziente
        patient_stats = df_rec.groupby('patient_id').agg(
            bag_size=('tissue_type', 'count'),
            num_unique_tissues=('tissue_type', lambda x: len(set(x)))
        ).reset_index()

        df_rec = df_rec.merge(patient_stats, on='patient_id', how='left')

        # 1. Aggregazione Globale (Tutti i pazienti)
        df_agg_all = df_rec.groupby('tissue_type').agg(
            mean_attention_all=('attention_weight', 'mean'),
            sample_count_all=('attention_weight', 'count')
        ).reset_index()

        # 2. Aggregazione solo per Pazienti Multi-Tessuto Misti (Competizione non viziata da N=1)
        df_mixed = df_rec[df_rec['num_unique_tissues'] > 1]
        if not df_mixed.empty:
            df_agg_mixed = df_mixed.groupby('tissue_type').agg(
                mean_attention_mixed=('attention_weight', 'mean'),
                std_attention_mixed=('attention_weight', 'std'),
                median_attention_mixed=('attention_weight', 'median'),
                sample_count_mixed=('attention_weight', 'count')
            ).reset_index()
            df_agg = df_agg_all.merge(df_agg_mixed, on='tissue_type', how='left').fillna(0.0)
        else:
            df_agg = df_agg_all
            df_agg['mean_attention_mixed'] = df_agg['mean_attention_all']
            df_agg['std_attention_mixed'] = 0.0
            df_agg['median_attention_mixed'] = df_agg['mean_attention_all']
            df_agg['sample_count_mixed'] = 0

        # Percentuali relative
        tot_all = df_agg['mean_attention_all'].sum()
        df_agg['pct_attention_all'] = (df_agg['mean_attention_all'] / tot_all * 100) if tot_all > 0 else 0.0

        tot_mixed = df_agg['mean_attention_mixed'].sum()
        df_agg['pct_attention_mixed'] = (df_agg['mean_attention_mixed'] / tot_mixed * 100) if tot_mixed > 0 else 0.0

        df_agg = df_agg.sort_values(by='mean_attention_mixed', ascending=False)
    else:
        df_agg = pd.DataFrame()

    out_path = os.path.join(output_dir, FN_MIL_TISSUE_ATTENTION_AGGREGATION)
    df_agg.to_csv(out_path, sep='\t', index=False)

    raw_out_path = os.path.join(output_dir, "mil_patient_tissue_attention_raw.tsv")
    df_rec.to_csv(raw_out_path, sep='\t', index=False)

    return df_agg, df_rec