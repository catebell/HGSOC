import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import fisher_exact, mannwhitneyu

import config_file_tree

# =============================================================================
# 1. ENVIRONMENT & MODULE PATH SETUP
# =============================================================================
TRAINING_DIR = os.path.join(config_file_tree.ROOT_DIR, "5_training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset
from utils_train_eval import pre_train_imputation_encoding

try:
    import train_functions  # Required for unpickling SelectiveNodeScaler inside fold artifacts
    from train_functions import _compute_bag_pool_weights
except ImportError:
    def _compute_bag_pool_weights(X_bag, pooling_mode="softmax", tau=0.20):
        """Fallback bag pooling weights when train_functions is unavailable."""
        n_samples, n_dims = X_bag.shape
        if n_samples == 1 or pooling_mode == "mean":
            weights = np.full((n_samples, n_dims), 1.0 / n_samples, dtype=np.float64)
        elif pooling_mode == "softmax":
            scaled = X_bag / max(float(tau), 1e-6)
            scaled -= scaled.max(axis=0, keepdims=True)
            exp_x = np.exp(scaled)
            weights = exp_x / np.clip(exp_x.sum(axis=0, keepdims=True), 1e-12, None)
        elif pooling_mode == "top2":
            k = min(2, n_samples)
            weights = np.zeros((n_samples, n_dims), dtype=np.float64)
            top_idx = np.argpartition(-X_bag, kth=k - 1, axis=0)[:k, :]
            weights[top_idx, np.arange(n_dims)[None, :]] = 1.0 / k
        else:
            weights = np.zeros((n_samples, n_dims), dtype=np.float64)
            weights[X_bag.argmax(axis=0), np.arange(n_dims)] = 1.0
        return np.sum(weights * X_bag, axis=0), weights

from utils_explainability import (
    ProgressTracker,
    tracked_tqdm,
    to_rel_path,
    load_gene_mappings,
    get_gene_name,
    group_multicol_features,
    aggregate_multicol_features_shap,
    plot_shap_summary_beeswarm,
    plot_importance_barplot,
    plot_gene_feature_boxplots,
    plot_gene_feature_attention_heatmap,
    plot_interactive_gene_network,
    plot_static_gene_network,
    run_posthoc_pathway_enrichment,
    load_model_training_config,
    get_all_shap_tsv_filename,
    get_all_shap_png_filename,
    get_gene_shap_tsv_filename,
    get_gene_shap_png_filename,
    get_gene_shap_npz_filename,
    get_gene_attention_map_tsv_filename,
    get_edge_attention_tsv_filename,
    get_net_interactive_html_filename,
    get_net_static_png_filename,
    get_heatmap_png_filename,
    get_importance_tsv_filename,
    get_importance_png_filename,
    compute_pca_umap_metrics,
    decode_class_label,
    generate_filtered_gene_shap_beeswarm,
    FN_MIL_TISSUE_ATTENTION_AGGREGATION,
    FN_MIL_TISSUE_ATTENTION_BARPLOT_PNG,
    resolve_experiment_data_dirs,
    compute_silhouette_only,
    _plot_sample_embedding_grid,
    _plot_patient_target_scatter,
    parse_multi_choice,
    resolve_step_sequence, evaluate_and_plot_sensitivity_specificity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 2. DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_RUN_FOLDER = "hybrid_LinearMIL_platRes_tissueDeconf_noClin_PDS_19_11_2026"
DEFAULT_STEPS = ["all"]
DEFAULT_COHORTS = ["all"]
DEFAULT_CLASSIFIERS = ["all"]
GENERATE_PLOTS = True

ALL_COHORTS = ["oof", "test"]
ALL_CLASSIFIERS = ["elasticnet", "plsda"]
CLASSIFIER_DIR_NAMES = {
    "elasticnet": "ElasticNet",
    "plsda": "PLS-DA",
}
STEP_WEIGHTS = {
    "metrics": 5,
    "graph": 20,
    "importance": 10,
    "shap": 30,
    "enrichment": 15,
    "tissue_attn": 5,
    "pca": 15,
}


# =============================================================================
# 3. EXPERIMENT PATH RESOLUTION
# =============================================================================
def _resolve_experiment_dirs(run_or_mil_arg, gat_arg=None):
    """Resolve (mil_models_dir, gat_models_dir) from an experiment folder name or explicit path."""
    def _to_abs(path_str):
        if not path_str:
            return None
        if os.path.exists(path_str):
            return os.path.abspath(path_str)
        return os.path.abspath(os.path.join(config_file_tree.TRAINED_MODELS_DIR, path_str))

    base_path = _to_abs(run_or_mil_arg)
    if os.path.isfile(base_path):
        base_path = os.path.dirname(base_path)

    mil_sub = os.path.join(base_path, "LinearMIL")
    mil_dir = mil_sub if os.path.isdir(mil_sub) else base_path

    if gat_arg:
        gat_dir = _to_abs(gat_arg)
    else:
        sibling_gat = os.path.join(os.path.dirname(mil_dir), "GAT")
        if os.path.isdir(sibling_gat) and any(f.endswith((".pt", ".joblib")) for f in os.listdir(sibling_gat)):
            gat_dir = sibling_gat
        else:
            gat_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)

    return mil_dir, gat_dir


# =============================================================================
# 4. LINEARIZATION, TISSUE DE-CONFOUNDING & POOLING HELPERS
# =============================================================================
def _get_effective_linear_weights(artifact, classifier_type="elasticnet"):
    """Extract effective linear weights (beta_raw, intercept_raw) in unstandardized patient space."""
    if classifier_type == "plsda":
        scaler = artifact["scaler_pls"]
        pls = artifact["pls"]
        clf = artifact["pls_clf"]
        w_std = pls.x_rotations_ @ clf.coef_.ravel()
    else:
        pipe = artifact["best_enet"]
        scaler = pipe.named_steps["scaler"]
        clf = pipe.named_steps["clf"]
        w_std = clf.coef_.ravel()

    beta_raw = w_std / np.clip(scaler.scale_, 1e-8, None)
    intercept_raw = float(clf.intercept_[0] - np.dot(beta_raw, scaler.mean_))
    return beta_raw, intercept_raw


def _compute_tissue_deconf_params(df_samp_train_raw, emb_cols):
    """Compute fold-specific tissue centroid statistics (centering only)."""
    return {
        "global_mean": df_samp_train_raw[emb_cols].mean(axis=0).values,
        "tissue_means": df_samp_train_raw.groupby("tissue_label")[emb_cols].mean(),
    }


def _deconfound_with_jacobian(h_raw, tissue_label, deconf_params):
    """Apply tissue centroid centering followed by L2 normalization, returning (h_tilde, ||u||_2)."""
    tissue_means = deconf_params["tissue_means"]
    mu = tissue_means.loc[tissue_label].values if tissue_label in tissue_means.index else deconf_params["global_mean"]
    u = h_raw - mu
    u_norm = max(float(np.linalg.norm(u)), 1e-6)
    return u / u_norm, u_norm


def _format_pool_label(pool_mode):
    """Return a human-readable label for the secondary bag pooling branch."""
    if pool_mode == "softmax":
        return "GAT_SoftMax_Pool (64D)"
    if pool_mode == "top2":
        return "GAT_Top2_Pool (64D)"
    return "GAT_Max_Pool (64D)"


def _load_fold_artifacts(mil_models_dir):
    """Load pre-saved 5-fold artifacts from fold_models/."""
    fold_models_dir = os.path.join(mil_models_dir, "fold_models")
    paths = [os.path.join(fold_models_dir, f"fold_{k}_artifacts.joblib") for k in range(1, 6)]
    if not all(os.path.exists(p) for p in paths):
        raise FileNotFoundError(f"Pre-saved 5-fold artifacts not found in: {to_rel_path(fold_models_dir)}")
    logging.info(f"Loaded 5-fold artifacts from: {to_rel_path(fold_models_dir)}")
    return [joblib.load(p) for p in paths]


def _build_val_pid_to_fold_map(fold_artifacts, mil_models_dir):
    """Map each patient_id to its held-out validation fold index (0..4)."""
    val_pid_to_fold = {}
    for f_idx, art in enumerate(fold_artifacts):
        for vp in art.get("val_patient_ids", []):
            val_pid_to_fold[str(vp).strip()] = f_idx

    if not val_pid_to_fold:
        oof_tsv = os.path.join(mil_models_dir, "oof_patient_predictions.tsv")
        if os.path.exists(oof_tsv):
            df_oof_ref = pd.read_csv(oof_tsv, sep="\t")
            for _, row in df_oof_ref.iterrows():
                val_pid_to_fold[str(row["patient_id"]).strip()] = int(row["fold"]) - 1
    return val_pid_to_fold


# =============================================================================
# 5. SHARED GRAPH & GAT EMBEDDING CACHE
# =============================================================================
class GraphFeatureCache:
    """Cache invariant sample-level tensors and GAT forward outputs across runs."""

    def __init__(self, model_gat, fold_artifacts, num_unique_genes, device):
        self.model_gat = model_gat
        self.fold_artifacts = fold_artifacts
        self.ref_node_scaler = fold_artifacts[0]["node_scaler"]
        self.num_unique_genes = num_unique_genes
        self.device = device

        self._scaled_x_ref = {}
        self._h_raw_by_fold = {}
        self._edge_attn_by_sample = {}
        self._channel_gene_scores = {}

    def get_ref_scaled_x(self, graph):
        sid = str(graph.sample_id)
        if sid not in self._scaled_x_ref:
            self._scaled_x_ref[sid] = self.ref_node_scaler.transform(graph.x_raw.numpy()).astype(np.float32)
        return self._scaled_x_ref[sid]

    def precompute_fold_embeddings(self, graphs, fold_indices, batch_size=32):
        self.model_gat.eval()
        for f_idx in fold_indices:
            missing_graphs = [g for g in graphs if (f_idx, str(g.sample_id)) not in self._h_raw_by_fold]
            if not missing_graphs:
                continue

            node_scaler = self.fold_artifacts[f_idx]["node_scaler"]
            with torch.no_grad():
                for start in range(0, len(missing_graphs), batch_size):
                    chunk = missing_graphs[start:start + batch_size]
                    data_list = []
                    for g in chunk:
                        g_clone = g.clone()
                        g_clone.x = torch.tensor(node_scaler.transform(g.x_raw.numpy()), dtype=torch.float32)
                        data_list.append(g_clone)

                    batch = Batch.from_data_list(data_list).to(self.device)
                    _, s_emb = self.model_gat(batch.x, batch.edge_index, batch.batch)
                    s_emb_np = s_emb.cpu().numpy()

                    for idx_in_chunk, g in enumerate(chunk):
                        self._h_raw_by_fold[(f_idx, str(g.sample_id))] = s_emb_np[idx_in_chunk]

    def get_h_raw(self, fold_idx, graph):
        key = (fold_idx, str(graph.sample_id))
        if key not in self._h_raw_by_fold:
            self.precompute_fold_embeddings([graph], [fold_idx], batch_size=1)
        return self._h_raw_by_fold[key]

    def get_sample_edge_attention(self, graph):
        sid = str(graph.sample_id)
        if sid in self._edge_attn_by_sample:
            return self._edge_attn_by_sample[sid]

        x_np = self.get_ref_scaled_x(graph)
        x_scaled = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        edge_idx = graph.edge_index.to(self.device)

        with torch.no_grad():
            node_indices = torch.arange(x_scaled.size(0), device=self.device) % self.num_unique_genes
            gene_emb = self.model_gat.gene_embedding(node_indices)
            x_in_conv = torch.cat([x_scaled, gene_emb], dim=-1)
            _, (e_idx_out, alpha) = self.model_gat.conv1(x_in_conv, edge_idx, return_attention_weights=True)

            mean_alpha = alpha.mean(dim=1).cpu().numpy()
            src_n = (e_idx_out[0] % self.num_unique_genes).cpu().numpy()
            dst_n = (e_idx_out[1] % self.num_unique_genes).cpu().numpy()

        edge_dict = {}
        for e_i in range(len(mean_alpha)):
            u, v = int(src_n[e_i]), int(dst_n[e_i])
            if u != v:
                ek = (u, v) if u < v else (v, u)
                edge_dict[ek] = edge_dict.get(ek, 0.0) + float(mean_alpha[e_i])

        self._edge_attn_by_sample[sid] = edge_dict
        return edge_dict

    def get_cohort_channel_gene_importance(self, cohort, sample_graphs, max_samples=40):
        if cohort in self._channel_gene_scores:
            return self._channel_gene_scores[cohort]

        subset = sample_graphs[:min(max_samples, len(sample_graphs))]
        ch_gene_mat = np.zeros((64, self.num_unique_genes), dtype=np.float64)

        for g in subset:
            x_np = self.get_ref_scaled_x(g)
            x_sc = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
            b_vec = torch.zeros(x_sc.size(0), dtype=torch.long, device=self.device)
            _, s_emb = self.model_gat(x_sc, g.edge_index.to(self.device), b_vec)

            for ch in range(64):
                grad_ch = torch.autograd.grad(s_emb[0, ch], x_sc, retain_graph=(ch < 63))[0].detach().cpu().numpy()
                ch_gene_mat[ch] += np.abs(grad_ch * x_np).sum(axis=1)

        self._channel_gene_scores[cohort] = ch_gene_mat
        return ch_gene_mat


# =============================================================================
# 6. PATIENT REPRESENTATIONS, LINEAR SHAP & EXACT JACOBIAN SENSITIVITIES
# =============================================================================
def _compute_patient_representations_and_sensitivities(
    patient_ids_sorted,
    graphs_by_patient,
    patient_target_map,
    df_clinical_clean,
    patient_target_col,
    fold_artifacts,
    val_pid_to_fold,
    graph_cache,
    label_encoder_sample,
    mode="oof",
):
    """Compute patient bag representations, linear SHAP values, and biopsy sensitivities."""
    active_cols_ref = fold_artifacts[0]["active_model_cols"]
    clinical_cols_ref = fold_artifacts[0]["clinical_feat_cols"]

    n_pat = len(patient_ids_sorted)
    d_tot = len(active_cols_ref)
    X_pat_mat = np.zeros((n_pat, d_tot), dtype=np.float64)
    SHAP_pat_mat = np.zeros((n_pat, d_tot), dtype=np.float64)
    y_pat_arr = np.array([patient_target_map[pid] for pid in patient_ids_sorted], dtype=int)
    patient_sample_sensitivities = {pid: [] for pid in patient_ids_sorted}

    all_cohort_graphs = [g for pid in patient_ids_sorted for g in graphs_by_patient[pid]]
    graph_cache.precompute_fold_embeddings(all_cohort_graphs, list(range(len(fold_artifacts))))

    for p_idx, pid in enumerate(patient_ids_sorted):
        p_graphs = graphs_by_patient[pid]
        f_indices = [val_pid_to_fold[pid]] if (mode == "oof" and pid in val_pid_to_fold) else list(range(len(fold_artifacts)))
        inv_f = 1.0 / len(f_indices)

        per_sample_v_accum = [np.zeros(64, dtype=np.float64) for _ in p_graphs]
        per_sample_h_raw_last = [None] * len(p_graphs)
        per_sample_h_tilde_last = [None] * len(p_graphs)

        for f_idx in f_indices:
            art = fold_artifacts[f_idx]
            deconf_params = art["deconf_params"]
            beta_raw = art["beta_raw"]
            x_train_mean = art["X_train_mean"]
            pool_mode = art.get("pooling_mode", "softmax")
            pool_tau = art.get("tau", 0.20)

            h_raw_list, h_tilde_list, unorm_list = [], [], []
            for g in p_graphs:
                h_raw = graph_cache.get_h_raw(f_idx, g)
                h_tilde, u_norm = _deconfound_with_jacobian(h_raw, int(g.y.item()), deconf_params)
                h_raw_list.append(h_raw)
                h_tilde_list.append(h_tilde)
                unorm_list.append(u_norm)

            H_tilde = np.vstack(h_tilde_list)
            mean_vec = H_tilde.mean(axis=0)
            max_vec, W_pool = _compute_bag_pool_weights(H_tilde, pooling_mode=pool_mode, tau=pool_tau)

            x_vec = np.concatenate([mean_vec, max_vec])
            if clinical_cols_ref and art["clin_prep"] is not None:
                df_c_row = df_clinical_clean[df_clinical_clean["patient_id"] == pid].copy()
                cp = art["clin_prep"]
                df_c_proc = pre_train_imputation_encoding(
                    df_c_row,
                    task="classification",
                    target_col=patient_target_col,
                    is_train=False,
                    encoders=cp["encoders"],
                    scalers=cp["scalers"],
                    imputers=cp["imputers"],
                    categories=cp["categories"],
                )
                c_vals = df_c_proc[clinical_cols_ref].values.ravel().astype(np.float64)
                x_vec = np.concatenate([x_vec, c_vals])

            X_pat_mat[p_idx] += x_vec * inv_f
            SHAP_pat_mat[p_idx] += (beta_raw * (x_vec - x_train_mean)) * inv_f

            beta_mean = beta_raw[:64]
            beta_max = beta_raw[64:128]
            n_bag = float(len(p_graphs))

            for j in range(len(p_graphs)):
                a_j = (beta_mean / n_bag) + (beta_max * W_pool[j])
                h_t = h_tilde_list[j]
                grad_u = (a_j - np.dot(a_j, h_t) * h_t) / unorm_list[j]
                per_sample_v_accum[j] += grad_u * inv_f
                per_sample_h_raw_last[j] = h_raw_list[j]
                per_sample_h_tilde_last[j] = h_t

        for j, g in enumerate(p_graphs):
            t_name = decode_class_label(int(g.y.item()), label_encoder_sample)
            patient_sample_sensitivities[pid].append(
                (g, per_sample_v_accum[j], per_sample_h_raw_last[j], per_sample_h_tilde_last[j], t_name)
            )

    return X_pat_mat, SHAP_pat_mat, y_pat_arr, patient_sample_sensitivities


def _compute_patient_gene_shap_tensors(
    patient_ids_sorted,
    patient_sample_sensitivities,
    graph_cache,
    model_gat,
    num_unique_genes,
    num_features,
    progress_desc="Gene-Level Gradient x Input SHAP",
):
    """Compute patient-level gene x modality attribution tensors via backpropagation through GAT."""
    n_pat = len(patient_ids_sorted)
    gene_shap_mats = [np.zeros((n_pat, num_unique_genes), dtype=np.float32) for _ in range(num_features)]
    gene_val_mats = [np.zeros((n_pat, num_unique_genes), dtype=np.float32) for _ in range(num_features)]

    model_gat.eval()
    for p_idx, pid in tracked_tqdm(list(enumerate(patient_ids_sorted)), desc=progress_desc):
        entries = patient_sample_sensitivities[pid]
        if not entries:
            continue
        inv_bag = 1.0 / float(len(entries))

        for g, v_j, _, _, _ in entries:
            x_np = graph_cache.get_ref_scaled_x(g)
            x_scaled = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
            b_vec = torch.zeros(x_scaled.size(0), dtype=torch.long, device=device)
            v_tensor = torch.tensor(v_j, dtype=torch.float32, device=device)

            _, s_emb = model_gat(x_scaled, g.edge_index.to(device), b_vec)
            score = torch.sum(s_emb.squeeze(0) * v_tensor)
            grad_x = torch.autograd.grad(score, x_scaled)[0].detach().cpu().numpy()

            node_shap = grad_x * x_np
            for f_i in range(num_features):
                gene_shap_mats[f_i][p_idx] += (node_shap[:, f_i] * inv_bag).astype(np.float32)
                gene_val_mats[f_i][p_idx] += (x_np[:, f_i] * inv_bag).astype(np.float32)

    return gene_shap_mats, gene_val_mats


# =============================================================================
# 7. EXPLAINABILITY STEP RUNNERS
# =============================================================================
def run_step_graph(
    output_dir,
    patient_ids_sorted,
    graphs_by_patient,
    y_pat_arr,
    patient_sample_sensitivities,
    gene_shap_mats,
    graph_cache,
    gene_map_dict,
    num_unique_genes,
    feature_names,
    patient_class_names,
):
    """Step 1: Gene saliency map, PPI edge attention networks, heatmap & top-5 boxplots."""
    sal_dir = os.path.join(output_dir, "gene_saliency_attention")
    net_dir = os.path.join(output_dir, "gene_networks")
    os.makedirs(sal_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)

    abs_grad_mean = np.stack([np.abs(m).mean(axis=0) for m in gene_shap_mats], axis=1)
    signed_grad_mean = np.stack([m.mean(axis=0) for m in gene_shap_mats], axis=1)
    max_val = abs_grad_mean.max() if abs_grad_mean.max() > 0 else 1.0
    norm_sal = abs_grad_mean / max_val

    sal_rows = []
    for idx in range(num_unique_genes):
        ensg_id, symbol = get_gene_name(idx, gene_map_dict)
        row = {
            "gene_id": ensg_id,
            "gene_name": symbol,
            "saliency_score": float(norm_sal[idx].mean()),
            "signed_resistance_score": float(signed_grad_mean[idx].sum()),
        }
        for f_i, f_n in enumerate(feature_names):
            row[f_n] = float(norm_sal[idx, f_i])
        sal_rows.append(row)

    df_sal = pd.DataFrame(sal_rows).sort_values(by="saliency_score", ascending=False)
    sal_path = os.path.join(sal_dir, get_gene_attention_map_tsv_filename("mil"))
    df_sal.to_csv(sal_path, sep="\t", index=False)

    edge_attention_sum = {}
    total_patients_used = 0
    for pid in patient_ids_sorted:
        entries = patient_sample_sensitivities[pid]
        if not entries:
            continue
        sample_scores = [abs(float(np.dot(v_j, h_raw_j))) for (_, v_j, h_raw_j, _, _) in entries]
        tot_score = sum(sample_scores) + 1e-8

        for idx_e, (g, _, _, _, _) in enumerate(entries):
            s_weight = sample_scores[idx_e] / tot_score
            for ek, att_val in graph_cache.get_sample_edge_attention(g).items():
                edge_attention_sum[ek] = edge_attention_sum.get(ek, 0.0) + att_val * s_weight
        total_patients_used += 1

    inv_patients = 1.0 / max(1, total_patients_used)
    edge_rows = [
        {
            "gene_id_a": get_gene_name(u, gene_map_dict)[0],
            "gene_symbol_a": get_gene_name(u, gene_map_dict)[1],
            "gene_id_b": get_gene_name(v, gene_map_dict)[0],
            "gene_symbol_b": get_gene_name(v, gene_map_dict)[1],
            "interaction": f"{get_gene_name(u, gene_map_dict)[1]} - {get_gene_name(v, gene_map_dict)[1]}",
            "attention_weight": float(tot_att * inv_patients),
        }
        for (u, v), tot_att in edge_attention_sum.items()
    ]
    df_edges = pd.DataFrame(edge_rows).sort_values(by="attention_weight", ascending=False)
    edge_tsv_path = os.path.join(net_dir, get_edge_attention_tsv_filename("mil"))
    df_edges.to_csv(edge_tsv_path, sep="\t", index=False)

    if GENERATE_PLOTS:
        plot_interactive_gene_network(
            tsv_path=edge_tsv_path,
            output_html=os.path.join(net_dir, get_net_interactive_html_filename("mil")),
            top_n=100,
            saliency_tsv_path=sal_path,
        )
        plot_static_gene_network(
            tsv_path=edge_tsv_path,
            output_png=os.path.join(net_dir, get_net_static_png_filename("mil")),
            top_n=100,
        )
        plot_gene_feature_attention_heatmap(
            tsv_path=sal_path,
            output_png=os.path.join(sal_dir, get_heatmap_png_filename("mil")),
            feature_names=feature_names,
            top_n=25,
        )

        top5 = df_sal.head(5)
        top5_dict = dict(zip(top5["gene_id"], top5["gene_name"]))
        ensg_to_idx = {v[0]: k for k, v in gene_map_dict.items()}
        box_records = []
        for p_i, pid in enumerate(patient_ids_sorted):
            mean_x = np.mean([graph_cache.get_ref_scaled_x(g) for g in graphs_by_patient[pid]], axis=0)
            row_box = {"label": int(y_pat_arr[p_i])}
            for gid in top5_dict:
                row_box[gid] = mean_x[ensg_to_idx[gid]] if gid in ensg_to_idx else np.nan
            box_records.append(row_box)

        plot_gene_feature_boxplots(
            box_records,
            top5_dict,
            class_names=patient_class_names,
            output_dir=sal_dir,
            prefix="mil",
            feature_names=feature_names,
        )


def run_step_importance(
    output_dir,
    patient_ids_sorted,
    X_pat_mat,
    y_pat_arr,
    fold_artifacts,
    val_pid_to_fold,
    clinical_cols_ref,
    classifier_label,
    cohort,
):
    """Step 2: Permutation feature importance (ROC-AUC and PR-AUC drop)."""
    imp_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(imp_dir, exist_ok=True)
    out_imp_path = os.path.join(imp_dir, get_importance_tsv_filename("mil"))

    pool_label = _format_pool_label(fold_artifacts[0].get("pooling_mode", "softmax"))
    groups = [
        {"name": "GAT_Mean_Pool (64D)", "indices": list(range(0, 64))},
        {"name": pool_label, "indices": list(range(64, 128))},
    ]
    for grp in group_multicol_features(clinical_cols_ref):
        groups.append({"name": grp["name"], "indices": [128 + idx for idx in grp["indices"]]})

    mean_beta = np.mean([art["beta_raw"] for art in fold_artifacts], axis=0)
    mean_intercept = float(np.mean([art["intercept_raw"] for art in fold_artifacts]))

    n_pat = len(patient_ids_sorted)
    B_pat = np.zeros_like(X_pat_mat)
    b0_pat = np.zeros(n_pat, dtype=np.float64)
    for p_idx, pid in enumerate(patient_ids_sorted):
        if cohort == "oof" and pid in val_pid_to_fold:
            art = fold_artifacts[val_pid_to_fold[pid]]
            B_pat[p_idx] = art["beta_raw"]
            b0_pat[p_idx] = art["intercept_raw"]
        else:
            B_pat[p_idx] = mean_beta
            b0_pat[p_idx] = mean_intercept

    def _predict_probs(X_in):
        logits = b0_pat + np.sum(X_in * B_pat, axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    base_probs = _predict_probs(X_pat_mat)
    base_auc = roc_auc_score(y_pat_arr, base_probs)
    base_pr_auc = average_precision_score(y_pat_arr, base_probs)

    rng = np.random.default_rng(42)
    imp_rows = []
    for grp in groups:
        auc_drops, pr_drops = [], []
        for _ in range(25):
            X_perm = X_pat_mat.copy()
            X_perm[:, grp["indices"]] = X_pat_mat[rng.permutation(n_pat)][:, grp["indices"]]
            p_perm = _predict_probs(X_perm)
            auc_drops.append(base_auc - roc_auc_score(y_pat_arr, p_perm))
            pr_drops.append(base_pr_auc - average_precision_score(y_pat_arr, p_perm))

        imp_rows.append({
            "feature": grp["name"],
            "importance": float(np.mean(auc_drops)),
            "importance_std": float(np.std(auc_drops)),
            "pr_auc_drop": float(np.mean(pr_drops)),
            "pr_auc_drop_std": float(np.std(pr_drops)),
        })

    df_imp = pd.DataFrame(imp_rows).sort_values(by="importance", ascending=False)
    df_imp.to_csv(out_imp_path, sep="\t", index=False)

    if GENERATE_PLOTS:
        plot_importance_barplot(
            df_imp,
            x_col="importance",
            y_col="feature",
            title=f"Feature Importance ({classifier_label}, {cohort.upper()} ROC-AUC Drop)",
            output_path=os.path.join(imp_dir, get_importance_png_filename("mil")),
            orientation="horizontal",
        )


def run_step_shap(
    output_dir,
    patient_ids_sorted,
    X_pat_mat,
    SHAP_pat_mat,
    gene_shap_mats,
    gene_val_mats,
    fold_artifacts,
    genomic_cols_ref,
    clinical_cols_ref,
    target_graphs,
    graph_cache,
    gene_map_dict,
    num_unique_genes,
    feature_names,
    classifier_label,
    cohort,
):
    """Step 3: Exact linear SHAP (patient-level, 128D latent space, and gene-level matrices)."""
    save_dir = os.path.join(output_dir, "shap_analysis")
    os.makedirs(save_dir, exist_ok=True)
    pool_label = _format_pool_label(fold_artifacts[0].get("pooling_mode", "softmax"))

    combined_shap = np.hstack([SHAP_pat_mat[:, :64].sum(axis=1, keepdims=True), SHAP_pat_mat[:, 64:128].sum(axis=1, keepdims=True)])
    combined_vals = np.hstack([X_pat_mat[:, :64].mean(axis=1, keepdims=True), X_pat_mat[:, 64:128].mean(axis=1, keepdims=True)])
    combined_names = ["GAT_Mean_Pool (64D)", pool_label]

    if clinical_cols_ref:
        agg_c_shap, agg_c_vals, agg_c_names = aggregate_multicol_features_shap(
            SHAP_pat_mat[:, 128:], X_pat_mat[:, 128:], clinical_cols_ref
        )
        combined_shap = np.hstack([combined_shap, agg_c_shap])
        combined_vals = np.hstack([combined_vals, agg_c_vals])
        combined_names += agg_c_names

    df_all_shap = pd.DataFrame({
        "feature": combined_names,
        "shap_importance": np.abs(combined_shap).mean(axis=0),
    }).sort_values(by="shap_importance", ascending=False)
    df_all_shap.to_csv(os.path.join(save_dir, get_all_shap_tsv_filename("mil")), sep="\t", index=False)

    if GENERATE_PLOTS:
        plot_shap_summary_beeswarm(
            shap_values=combined_shap,
            features=combined_vals,
            feature_names=combined_names,
            title=f"MIL Patient All-Features SHAP ({classifier_label}, {cohort.upper()})",
            output_path=os.path.join(save_dir, get_all_shap_png_filename("mil")),
        )

    betas_128 = np.array([art["beta_raw"][:128] for art in fold_artifacts])
    mean_beta_128 = betas_128.mean(axis=0)
    n_folds_nonzero = (np.abs(betas_128) > 1e-6).sum(axis=0)

    df_lat_shap = pd.DataFrame({
        "feature": genomic_cols_ref,
        "dim_idx_128": np.arange(128),
        "gat_channel": [d % 64 for d in range(128)],
        "pooling_type": ["mean_emb" if d < 64 else "max_emb" for d in range(128)],
        "mean_beta": mean_beta_128,
        "folds_active": n_folds_nonzero,
        "shap_importance": np.abs(SHAP_pat_mat[:, :128]).mean(axis=0),
        "mean_signed_shap": SHAP_pat_mat[:, :128].mean(axis=0),
    }).sort_values(by="shap_importance", ascending=False)
    df_lat_shap.drop(columns=["dim_idx_128"]).to_csv(
        os.path.join(save_dir, "mil_latent_dims_shap.tsv"), sep="\t", index=False
    )

    if GENERATE_PLOTS:
        top_lat_idx = df_lat_shap.head(20)["dim_idx_128"].values
        plot_shap_summary_beeswarm(
            shap_values=SHAP_pat_mat[:, top_lat_idx],
            features=X_pat_mat[:, top_lat_idx],
            feature_names=[genomic_cols_ref[i] for i in top_lat_idx],
            title=f"MIL Top 20 GAT Latent Dimensions SHAP ({classifier_label}, {cohort.upper()})",
            output_path=os.path.join(save_dir, "mil_latent_dims_shap_summary.png"),
        )

    active_dims_df = df_lat_shap[df_lat_shap["folds_active"] >= 2].head(15)
    if active_dims_df.empty:
        active_dims_df = df_lat_shap[df_lat_shap["folds_active"] >= 1].head(15)

    ch_gene_mat = graph_cache.get_cohort_channel_gene_importance(cohort, target_graphs, max_samples=40)
    dim_to_gene_rows = []
    for _, d_row in active_dims_df.iterrows():
        gat_ch = int(d_row["gat_channel"])
        top5_g_idx = np.argsort(-ch_gene_mat[gat_ch])[:5]
        top5_genes_str = ", ".join([get_gene_name(int(gi), gene_map_dict)[1] for gi in top5_g_idx])
        dim_to_gene_rows.append({
            "latent_feature": d_row["feature"],
            "pooling": d_row["pooling_type"],
            "gat_channel": gat_ch,
            "mean_linear_beta": float(d_row["mean_beta"]),
            "direction": "Resistance (yes)" if d_row["mean_beta"] > 0 else "Sensitivity (no)",
            "folds_active": f"{int(d_row['folds_active'])}/5",
            "mean_abs_shap": float(d_row["shap_importance"]),
            "top5_driving_genes_HGNC": top5_genes_str,
        })

    pd.DataFrame(dim_to_gene_rows).to_csv(
        os.path.join(save_dir, "nonzero_dims_to_genes.tsv"), sep="\t", index=False
    )

    all_gene_ids = [get_gene_name(g_idx, gene_map_dict)[0] for g_idx in range(num_unique_genes)]
    all_gene_symbols = [get_gene_name(g_idx, gene_map_dict)[1] for g_idx in range(num_unique_genes)]

    for f_i, feat_name in enumerate(feature_names):
        f_shap = gene_shap_mats[f_i]
        f_val = gene_val_mats[f_i]

        np.savez_compressed(
            os.path.join(save_dir, get_gene_shap_npz_filename("mil", feat_name)),
            shap_values=f_shap,
            features=f_val,
            gene_names=np.array(all_gene_symbols, dtype=object),
            gene_ids=np.array(all_gene_ids, dtype=object),
            patient_ids=np.array(patient_ids_sorted, dtype=object),
        )

        df_g_shap = pd.DataFrame({
            "gene_id": all_gene_ids,
            "gene_name": all_gene_symbols,
            "gene_idx": np.arange(num_unique_genes),
            "shap_importance": np.abs(f_shap).mean(axis=0).astype(float),
            "mean_signed_shap": f_shap.mean(axis=0).astype(float),
        }).sort_values(by="shap_importance", ascending=False)

        df_g_shap.drop(columns=["gene_idx"]).to_csv(
            os.path.join(save_dir, get_gene_shap_tsv_filename("mil", feat_name)), sep="\t", index=False
        )

        if GENERATE_PLOTS:
            top_df = df_g_shap.head(20)
            sel_idx = top_df["gene_idx"].values
            plot_shap_summary_beeswarm(
                shap_values=f_shap[:, sel_idx],
                features=f_val[:, sel_idx],
                feature_names=top_df["gene_name"].tolist(),
                title=f"MIL Top 20 Genes SHAP ({feat_name}, {classifier_label}, {cohort.upper()})",
                output_path=os.path.join(save_dir, get_gene_shap_png_filename("mil", feat_name)),
            )


def run_step_enrichment(output_dir, feature_names):
    """Step 4: Post-hoc directional ORA and Preranked Hallmark GSEA."""
    sal_path = os.path.join(output_dir, "gene_saliency_attention", get_gene_attention_map_tsv_filename("mil"))
    shap_paths = [
        os.path.join(output_dir, "shap_analysis", get_gene_shap_tsv_filename("mil", fn))
        for fn in feature_names
    ]
    run_posthoc_pathway_enrichment(
        saliency_tsv_path=sal_path,
        output_dir=output_dir,
        shap_tsv_paths=shap_paths,
        directional_ora=True,
    )


def run_step_tissue_attn(output_dir, patient_ids_sorted, patient_sample_sensitivities, classifier_label):
    """Step 5: Tissue decision contribution breakdown across single- and multi-tissue bags."""
    imp_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(imp_dir, exist_ok=True)

    rec_rows = []
    for pid in patient_ids_sorted:
        entries = patient_sample_sensitivities[pid]
        if not entries:
            continue
        raw_contribs = [abs(float(np.dot(v_j, h_raw_j))) for (_, v_j, h_raw_j, _, _) in entries]
        tot_c = sum(raw_contribs) + 1e-8
        unique_tissues = len({e[4] for e in entries})

        for idx_e, (_, _, _, _, t_name) in enumerate(entries):
            rec_rows.append({
                "patient_id": pid,
                "tissue_type": t_name,
                "attention_weight": raw_contribs[idx_e] / tot_c,
                "bag_size": len(entries),
                "num_unique_tissues": unique_tissues,
            })

    df_rec = pd.DataFrame(rec_rows)
    df_agg_all = df_rec.groupby("tissue_type").agg(
        mean_attention_all=("attention_weight", "mean"),
        sample_count_all=("attention_weight", "count"),
    ).reset_index()

    df_mixed = df_rec[df_rec["num_unique_tissues"] > 1]
    if not df_mixed.empty:
        df_agg_mixed = df_mixed.groupby("tissue_type").agg(
            mean_attention_mixed=("attention_weight", "mean"),
            std_attention_mixed=("attention_weight", "std"),
            median_attention_mixed=("attention_weight", "median"),
            sample_count_mixed=("attention_weight", "count"),
        ).reset_index()
        df_agg = df_agg_all.merge(df_agg_mixed, on="tissue_type", how="left").fillna(0.0)
    else:
        df_agg = df_agg_all.assign(
            mean_attention_mixed=df_agg_all["mean_attention_all"],
            std_attention_mixed=0.0,
            median_attention_mixed=df_agg_all["mean_attention_all"],
            sample_count_mixed=0,
        )

    df_agg["pct_attention_all"] = df_agg["mean_attention_all"] / max(df_agg["mean_attention_all"].sum(), 1e-8) * 100.0
    df_agg["pct_attention_mixed"] = df_agg["mean_attention_mixed"] / max(df_agg["mean_attention_mixed"].sum(), 1e-8) * 100.0
    df_agg = df_agg.sort_values(by="mean_attention_mixed", ascending=False)

    df_agg.to_csv(os.path.join(imp_dir, FN_MIL_TISSUE_ATTENTION_AGGREGATION), sep="\t", index=False)
    df_rec.to_csv(os.path.join(imp_dir, "mil_patient_tissue_attention_raw.tsv"), sep="\t", index=False)

    if GENERATE_PLOTS and not df_agg.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_agg, x="tissue_type", y="pct_attention_mixed", palette="Set2", hue="tissue_type", legend=False)
        plt.ylabel("Decision Contribution in Multi-Tissue Bags (%)")
        plt.xlabel("Tissue Type")
        plt.title(f"MIL Tissue Decision Share ({classifier_label})")
        plt.tight_layout()
        plt.savefig(os.path.join(imp_dir, FN_MIL_TISSUE_ATTENTION_BARPLOT_PNG), dpi=300)
        plt.close()


def run_step_pca(
    output_dir,
    patient_ids,
    X_pat_mat,
    y_pat_arr,
    patient_sample_sensitivities,
    fold_artifacts,
    val_pid_to_fold,
    label_encoder_patient,
    classifier_label,
    suffix="oof",
):
    """Step 6: Visualize PCA, UMAP, and PLS-DA projections for patient bags and de-confounded biopsies."""
    pca_dir = os.path.join(output_dir, "embeddings_pca")
    os.makedirs(pca_dir, exist_ok=True)

    Z_128 = X_pat_mat[:, :128]
    p_targets = np.array([decode_class_label(y, label_encoder_patient) for y in y_pat_arr])
    dom_tissues = np.array([
        pd.Series([e[4] for e in patient_sample_sensitivities[pid]]).mode()[0]
        if patient_sample_sensitivities[pid] else "Unknown"
        for pid in patient_ids
    ])

    pca_res, umap_res, var_exp, silh_target, X_scaled_z = compute_pca_umap_metrics(Z_128, p_targets)
    silh_tissue = compute_silhouette_only(X_scaled_z, dom_tissues)

    pls_coords = np.zeros((len(patient_ids), 2), dtype=np.float64)
    for p_idx, pid in enumerate(patient_ids):
        if suffix == "oof" and pid in val_pid_to_fold:
            art = fold_artifacts[val_pid_to_fold[pid]]
            pls_coords[p_idx] = art["pls"].transform(art["scaler_pls"].transform(X_pat_mat[p_idx:p_idx + 1]))[0]
        else:
            pls_coords[p_idx] = np.mean(
                [art["pls"].transform(art["scaler_pls"].transform(X_pat_mat[p_idx:p_idx + 1]))[0] for art in fold_artifacts],
                axis=0,
            )

    silh_pls = compute_silhouette_only(StandardScaler().fit_transform(pls_coords), p_targets)

    df_p = pd.DataFrame({
        "patient_id": patient_ids,
        "patient_target": p_targets,
        "dominant_tissue": dom_tissues,
        "PCA1": pca_res[:, 0],
        "PCA2": pca_res[:, 1],
        "PLS_LV1": pls_coords[:, 0],
        "PLS_LV2": pls_coords[:, 1],
    })
    if umap_res is not None:
        df_p["UMAP1"] = umap_res[:, 0]
        df_p["UMAP2"] = umap_res[:, 1]

    df_p.to_csv(os.path.join(pca_dir, f"patient_clusters_z128_{suffix}.tsv"), sep="\t", index=False)
    if not GENERATE_PLOTS:
        return

    n_rows = 2 if umap_res is not None else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(19, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    silh_t_str = f" | Silh: {silh_target:.3f}" if silh_target is not None else ""
    _plot_patient_target_scatter(
        df_p, "PCA1", "PCA2", axes[0, 0],
        f"Patient Bag z_i (R^128) PCA by Resistance\n(Var Exp: {var_exp:.1%}{silh_t_str})", size=95,
    )

    silh_d_str = f" | Silh: {silh_tissue:.3f}" if silh_tissue is not None else ""
    sns.scatterplot(data=df_p, x="PCA1", y="PCA2", hue="dominant_tissue", palette="Set1", s=95, edgecolor="k", ax=axes[0, 1])
    axes[0, 1].set_title(f"Patient Bag z_i (R^128) PCA by Dominant Tissue{silh_d_str}")

    silh_pls_str = f" | Silh: {silh_pls:.3f}" if silh_pls is not None else ""
    _plot_patient_target_scatter(
        df_p, "PLS_LV1", "PLS_LV2", axes[0, 2],
        f"PLS-DA Latent Space (LV1 vs LV2, {suffix.upper()}){silh_pls_str}", size=95,
    )
    for _, r in df_p.iterrows():
        axes[0, 2].text(r["PLS_LV1"], r["PLS_LV2"], str(r["patient_id"]), fontsize=7, alpha=0.8)

    if umap_res is not None:
        _plot_patient_target_scatter(df_p, "UMAP1", "UMAP2", axes[1, 0], "Patient Bag z_i (R^128) UMAP by Resistance", size=95)
        sns.scatterplot(data=df_p, x="UMAP1", y="UMAP2", hue="dominant_tissue", palette="Set1", s=95, edgecolor="k", ax=axes[1, 1])
        axes[1, 1].set_title("Patient Bag z_i (R^128) UMAP by Dominant Tissue")

        mean_beta = np.mean([art["beta_raw"][:128] for art in fold_artifacts], axis=0)
        top2_dims = np.argsort(-np.abs(mean_beta))[:2]
        dim_names = fold_artifacts[0]["genomic_feat_cols"]
        df_p["TopDim1"] = Z_128[:, top2_dims[0]]
        df_p["TopDim2"] = Z_128[:, top2_dims[1]]
        _plot_patient_target_scatter(
            df_p, "TopDim1", "TopDim2", axes[1, 2],
            f"Top-2 {classifier_label} Selected GAT Dimensions", size=95,
        )
        axes[1, 2].set_xlabel(f"{dim_names[top2_dims[0]]} (beta={mean_beta[top2_dims[0]]:+.3f})")
        axes[1, 2].set_ylabel(f"{dim_names[top2_dims[1]]} (beta={mean_beta[top2_dims[1]]:+.3f})")

    plt.tight_layout()
    out_p_path = os.path.join(pca_dir, f"mil_patient_embeddings_pca_{suffix}.png")
    plt.savefig(out_p_path, dpi=180, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved patient cluster plot ({classifier_label}, {suffix}) to: {to_rel_path(out_p_path)}")

    s_vecs, s_records = [], []
    for p_idx, pid in enumerate(patient_ids):
        for (_, _, _, h_tilde, t_name) in patient_sample_sensitivities[pid]:
            s_vecs.append(h_tilde)
            s_records.append({"patient_id": pid, "tissue_type": t_name, "patient_target": p_targets[p_idx]})

    if s_vecs:
        out_s_path = os.path.join(pca_dir, f"mil_sample_embeddings_pca_{suffix}.png")
        _plot_sample_embedding_grid(np.vstack(s_vecs), pd.DataFrame(s_records), "De-confounded Sample", out_s_path)
        logging.info(f"Saved sample cluster plot ({classifier_label}, {suffix}) to: {to_rel_path(out_s_path)}")


# =============================================================================
# 8. DIAGNOSTIC SUMMARY REPORT
# =============================================================================
def summarize_four_insights(mil_models_dir, classifier_key, cohort):
    """Print a concise post-run diagnostic summary under explained/<Classifier>/<cohort>/."""
    clf_label = CLASSIFIER_DIR_NAMES[classifier_key]
    output_dir = os.path.join(mil_models_dir, "explained", clf_label, cohort)
    shap_dir = os.path.join(output_dir, "shap_analysis")
    pca_dir = os.path.join(output_dir, "embeddings_pca")

    print(f"\n=========================================================================")
    print(f" DIAGNOSTIC SUMMARY: {clf_label} | Cohort: {cohort.upper()}")
    print(f"=========================================================================")

    oof_path = os.path.join(mil_models_dir, "oof_patient_predictions.tsv")
    if os.path.exists(oof_path):
        df_oof = pd.read_csv(oof_path, sep="\t")
        prob_col = "pls_prob_resistant" if classifier_key == "plsda" and "pls_prob_resistant" in df_oof.columns else "enet_prob_resistant"

        if prob_col in df_oof.columns:
            res_hc = df_oof[(df_oof["target_label_name"] == "yes") & (df_oof[prob_col] > 0.6)].copy()
            sen_hc = df_oof[(df_oof["target_label_name"] == "no") & (df_oof[prob_col] < 0.4)].copy()
            print(f"[1] High-Confidence OOF Patients ({prob_col}): Resistant (p>0.6) N={len(res_hc)} | Sensitive (p<0.4) N={len(sen_hc)}")

            if "is_hrd" in df_oof.columns and len(res_hc) > 0 and len(sen_hc) > 0:
                def _is_hrp(series):
                    return series.astype(str).str.strip().str.lower().isin(["0", "0.0", "no", "false"])

                hrp_res = int(_is_hrp(res_hc["is_hrd"]).sum())
                hrd_res = int(res_hc["is_hrd"].notna().sum()) - hrp_res
                hrp_sen = int(_is_hrp(sen_hc["is_hrd"]).sum())
                hrd_sen = int(sen_hc["is_hrd"].notna().sum()) - hrp_sen
                _, p_hrd = fisher_exact([[hrp_res, hrd_res], [hrp_sen, hrd_sen]])
                print(f"    HRP Rate -> Resistant: {hrp_res}/{hrp_res + hrd_res} | Sensitive: {hrp_sen}/{hrp_sen + hrd_sen} (Fisher p={p_hrd:.4f})")

            if "n_samples" in df_oof.columns and len(res_hc) > 0 and len(sen_hc) > 0:
                _, p_ns = mannwhitneyu(res_hc["n_samples"], sen_hc["n_samples"])
                print(f"    Mean Biopsies/Patient -> Resistant: {res_hc['n_samples'].mean():.2f} | Sensitive: {sen_hc['n_samples'].mean():.2f} (MWU p={p_ns:.4f})")

        clust_path = os.path.join(pca_dir, f"patient_clusters_z128_{cohort}.tsv")
        if os.path.exists(clust_path):
            df_cl = pd.read_csv(clust_path, sep="\t")
            cols_merge = [c for c in ["patient_id", "is_hrd", "Residual_tumor", "tissue_types", prob_col] if c in df_oof.columns]
            df_yes = df_cl[df_cl["patient_target"] == "yes"].merge(df_oof[cols_merge], on="patient_id", how="left").sort_values("PLS_LV2")
            print(f"\n[2] Resistant Patients ('yes') Ordered by PLS_LV2 ({cohort.upper()}):")
            print(df_yes.to_string(index=False))

    lat_path = os.path.join(shap_dir, "mil_latent_dims_shap.tsv")
    if os.path.exists(lat_path):
        df_lat = pd.read_csv(lat_path, sep="\t")
        df_lat_gen = df_lat[df_lat["feature"].str.startswith(("mean_emb_", "max_emb_"))].copy()
        df_lat_gen["pool_type"] = np.where(
            df_lat_gen["feature"].str.startswith("mean_emb_"),
            "mean_emb (Diffuse)",
            "max_emb (SoftMax/TopK)",
        )
        df_lat_gen["abs_beta"] = df_lat_gen["mean_beta"].abs()
        df_active = df_lat_gen[(df_lat_gen["folds_active"] >= 2) & (df_lat_gen["shap_importance"] > 0)]
        print(f"\n[3] Pooling Branch Dominance ({clf_label} | {cohort.upper()}, active >=2/5 folds):")
        print(df_active.groupby("pool_type").agg(
            n_dims=("feature", "count"),
            sum_abs_beta=("abs_beta", "sum"),
            sum_shap_imp=("shap_importance", "sum"),
            mean_shap_imp=("shap_importance", "mean"),
        ))

    target_genes = ["VWF", "COX7A1", "APLNR", "NDUFA1", "FN1", "HLA-G"]
    gshap_path = os.path.join(shap_dir, get_gene_shap_tsv_filename("mil", "gene_expression"))
    if os.path.exists(gshap_path):
        df_gs = pd.read_csv(gshap_path, sep="\t")
        df_gs["rank_shap"] = np.arange(1, len(df_gs) + 1)
        sub_gs = df_gs[df_gs["gene_name"].isin(target_genes)][["rank_shap", "gene_name", "shap_importance", "mean_signed_shap"]]
        print(f"\n[4] Candidate Genes Rank ({clf_label}, {cohort.upper()}):")
        print(sub_gs.to_string(index=False))

    npz_path = os.path.join(shap_dir, get_gene_shap_npz_filename("mil", "gene_expression"))
    out_beeswarm = os.path.join(shap_dir, "mil_selected_6genes_cohort_comparison_beeswarm.png")
    ok, n_matched = generate_filtered_gene_shap_beeswarm(
        npz_path=npz_path,
        target_genes=target_genes,
        title=f"De-confounded SHAP ({clf_label}, {cohort.upper()}): Candidate Genes",
        output_path=out_beeswarm,
    )
    if ok:
        print(f"    Saved filtered beeswarm plot ({n_matched}/6 genes) to: {to_rel_path(out_beeswarm)}")


# =============================================================================
# 9. MAIN ORCHESTRATION ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Frozen-GAT + De-confounded Bag MIL Explainability Pipeline")
    parser.add_argument(
        "--run_folder",
        type=str,
        default=os.path.join(DEFAULT_RUN_FOLDER, "LinearMIL"),
        help="Experiment folder name inside trained_models/ or full path",
    )
    parser.add_argument("--gat_model_path", type=str, default=None, help="Optional override path to GAT checkpoint directory")
    parser.add_argument(
        "--step",
        nargs="+",
        default=DEFAULT_STEPS,
        help="Steps to execute: 'all' or subset of [graph, importance, shap, enrichment, tissue_attn, pca]",
    )
    parser.add_argument(
        "--cohort",
        nargs="+",
        default=DEFAULT_COHORTS,
        help="Cohorts to explain: 'all' or subset of [oof, test]",
    )
    parser.add_argument(
        "--classifier",
        nargs="+",
        default=DEFAULT_CLASSIFIERS,
        help="Classifiers to explain: 'all' or subset of [elasticnet, plsda]",
    )
    args = parser.parse_args()

    selected_steps = resolve_step_sequence(args.step, model_type="linear_mil")
    selected_cohorts = parse_multi_choice(args.cohort, ALL_COHORTS)
    selected_classifiers = parse_multi_choice(args.classifier, ALL_CLASSIFIERS, alias_map={"pls-da": "plsda"})

    mil_models_dir, gat_models_dir = _resolve_experiment_dirs(args.run_folder, args.gat_model_path)
    mil_cfg = load_model_training_config(mil_models_dir)

    parent_dir_name = os.path.basename(os.path.dirname(mil_models_dir))
    if "withClin" in parent_dir_name:
        use_clinical = True
    elif "noClin" in parent_dir_name:
        use_clinical = False
    else:
        use_clinical = mil_cfg["USE_CLINICAL_DATA"]

    task = mil_cfg["TASK"]
    sample_target_col = mil_cfg["SAMPLE_TARGET_COL"]
    patient_target_col = mil_cfg["PATIENT_TARGET_COL"]
    use_samples_metadata = mil_cfg["USE_SAMPLES_METADATA"]
    node_feature_index = mil_cfg["NODE_FEATURE_INDEX"]
    node_feature_names = mil_cfg["NODE_FEATURE_NAMES"]
    feature_names = [node_feature_names[i] for i in node_feature_index if i < len(node_feature_names)]

    graphs_dir, split_dir, processed_samples_dir = resolve_experiment_data_dirs(mil_cfg)
    metadata_path = os.path.join(split_dir, "metadata_for_training.tsv")
    clinical_path = os.path.join(split_dir, "clinical_data_for_training.tsv")
    train_val_csv_path = os.path.join(split_dir, "samples_train_val.csv")
    test_csv_path = os.path.join(split_dir, "samples_test.csv")
    gene_mapping_file = os.path.join(processed_samples_dir, "gene_node_mapping.tsv")

    weighted_phases = {"init": 12}
    for clf_key in selected_classifiers:
        for cohort in selected_cohorts:
            for st_name in selected_steps:
                weighted_phases[f"{clf_key}_{cohort}_{st_name}"] = STEP_WEIGHTS[st_name]

    ProgressTracker.configure(weighted_phases)
    ProgressTracker.start_phase("init", "Loading datasets, frozen GAT model & 5-fold artifacts...")

    logging.info(f"MIL Dir: {to_rel_path(mil_models_dir)} | GAT Dir: {to_rel_path(gat_models_dir)}")
    logging.info(
        f"Classifiers: {[CLASSIFIER_DIR_NAMES[c] for c in selected_classifiers]} | "
        f"Cohorts: {[c.upper() for c in selected_cohorts]} | "
        f"Resolved Steps: {[s.upper() for s in selected_steps]} | USE_CLINICAL: {use_clinical}"
    )

    df_metadata = pd.read_csv(metadata_path, sep="\t")
    df_tv_samples = pd.read_csv(train_val_csv_path)
    df_test_samples = pd.read_csv(test_csv_path)

    df_metadata_tv = df_metadata[df_metadata["sample_id"].isin(df_tv_samples["sample_id"])].dropna(subset=[sample_target_col]).copy()
    df_metadata_test = df_metadata[df_metadata["sample_id"].isin(df_test_samples["sample_id"])].dropna(subset=[sample_target_col]).copy()

    cols_to_keep = ["patient_id", patient_target_col]
    if use_clinical:
        cols_to_keep += [c for c in config_file_tree.CLINICAL_FEATURES if c not in cols_to_keep]
    df_clinical = pd.read_csv(clinical_path, sep="\t", usecols=cols_to_keep)
    df_clinical["patient_id"] = df_clinical["patient_id"].astype(str).str.strip()
    df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).drop_duplicates(subset=["patient_id"]).copy()

    sample_le_path = os.path.join(gat_models_dir, f"label_encoder_{sample_target_col}.joblib")
    label_encoder_sample = (
        joblib.load(sample_le_path)
        if os.path.exists(sample_le_path)
        else LabelEncoder().fit(df_metadata[sample_target_col].dropna())
    )
    df_metadata_tv[sample_target_col] = label_encoder_sample.transform(df_metadata_tv[sample_target_col])
    df_metadata_test[sample_target_col] = label_encoder_sample.transform(df_metadata_test[sample_target_col])
    num_sample_classes = len(label_encoder_sample.classes_)

    patient_le_path = os.path.join(mil_models_dir, f"label_encoder_patient_{patient_target_col}.joblib")
    label_encoder_patient = (
        joblib.load(patient_le_path)
        if os.path.exists(patient_le_path)
        else LabelEncoder().fit(df_clinical_clean[patient_target_col])
    )
    df_clinical_clean[patient_target_col] = label_encoder_patient.transform(df_clinical_clean[patient_target_col])
    patient_class_names = [str(c) for c in label_encoder_patient.classes_]
    patient_target_map = df_clinical_clean.set_index("patient_id")[patient_target_col].to_dict()

    graphs_by_cohort = {}
    if "oof" in selected_cohorts:
        raw_tv_ds = PatientSampleDataset(graphs_dir=graphs_dir, samples_metadata_df=df_metadata_tv, task=task, target_col=sample_target_col)
        tv_graphs = [raw_tv_ds[i] for i in tracked_tqdm(range(len(raw_tv_ds)), desc="Loading Train/Val Graphs", start_frac=0.0, end_frac=0.6)]
        for g in tv_graphs:
            g.x_raw = g.x.clone()
        graphs_by_cohort["oof"] = tv_graphs

    if "test" in selected_cohorts:
        raw_test_ds = PatientSampleDataset(graphs_dir=graphs_dir, samples_metadata_df=df_metadata_test, task=task, target_col=sample_target_col)
        test_graphs = [raw_test_ds[i] for i in tracked_tqdm(range(len(raw_test_ds)), desc="Loading Test Graphs", start_frac=0.6, end_frac=0.9)]
        for g in test_graphs:
            g.x_raw = g.x.clone()
        graphs_by_cohort["test"] = test_graphs

    sample_graph_ref = next(iter(graphs_by_cohort.values()))[0]
    num_genes_features = sample_graph_ref.x.size(1)
    num_unique_genes = sample_graph_ref.x.size(0)
    gene_map_dict = load_gene_mappings(gene_mapping_file)

    fold_artifacts = _load_fold_artifacts(mil_models_dir)
    val_pid_to_fold = _build_val_pid_to_fold_map(fold_artifacts, mil_models_dir)

    pretrained_gat_path = os.path.join(
        gat_models_dir,
        f"best_GAT_{sample_target_col}_{'withMeta' if use_samples_metadata else 'noMeta'}.pt",
    )
    model_gat = GAT(
        num_node_features=num_genes_features,
        num_classes=num_sample_classes,
        num_nodes=num_unique_genes,
        num_sample_features=0,
    ).to(device)
    model_gat.load_state_dict(torch.load(pretrained_gat_path, map_location=device))
    model_gat.eval()

    emb_cols = [f"gat_emb_{d}" for d in range(64)]
    for art in fold_artifacts:
        art["deconf_params"] = _compute_tissue_deconf_params(art["df_samp_train_raw"], emb_cols)

    graph_cache = GraphFeatureCache(model_gat, fold_artifacts, num_unique_genes, device)

    cohort_bags = {}
    for cohort, g_list in graphs_by_cohort.items():
        grouped = {}
        for g in g_list:
            pid = str(g.patient_id).strip()
            if pid in patient_target_map:
                grouped.setdefault(pid, []).append(g)
        cohort_bags[cohort] = (grouped, sorted(grouped.keys()))

    genomic_cols_ref = fold_artifacts[0]["genomic_feat_cols"]
    clinical_cols_ref = fold_artifacts[0]["clinical_feat_cols"]
    need_gene_gradients = any(s in selected_steps for s in ("graph", "shap"))

    for clf_key in selected_classifiers:
        clf_label = CLASSIFIER_DIR_NAMES[clf_key]
        for art in fold_artifacts:
            beta_raw, intercept_raw = _get_effective_linear_weights(art, classifier_type=clf_key)
            art["beta_raw"] = beta_raw
            art["intercept_raw"] = intercept_raw

        for cohort in selected_cohorts:
            output_dir = os.path.join(mil_models_dir, "explained", clf_label, cohort)
            os.makedirs(output_dir, exist_ok=True)

            graphs_by_patient, patient_ids_sorted = cohort_bags[cohort]
            target_graphs = graphs_by_cohort[cohort]
            logging.info(
                f"--- Running [{clf_label} | {cohort.upper()}] -> {to_rel_path(output_dir)} "
                f"({len(patient_ids_sorted)} patients, {len(target_graphs)} biopsies) ---"
            )

            X_pat_mat, SHAP_pat_mat, y_pat_arr, patient_sample_sensitivities = (
                _compute_patient_representations_and_sensitivities(
                    patient_ids_sorted=patient_ids_sorted,
                    graphs_by_patient=graphs_by_patient,
                    patient_target_map=patient_target_map,
                    df_clinical_clean=df_clinical_clean,
                    patient_target_col=patient_target_col,
                    fold_artifacts=fold_artifacts,
                    val_pid_to_fold=val_pid_to_fold,
                    graph_cache=graph_cache,
                    label_encoder_sample=label_encoder_sample,
                    mode=cohort,
                )
            )

            gene_shap_mats, gene_val_mats = (None, None)
            if need_gene_gradients:
                gene_shap_mats, gene_val_mats = _compute_patient_gene_shap_tensors(
                    patient_ids_sorted=patient_ids_sorted,
                    patient_sample_sensitivities=patient_sample_sensitivities,
                    graph_cache=graph_cache,
                    model_gat=model_gat,
                    num_unique_genes=num_unique_genes,
                    num_features=len(feature_names),
                    progress_desc=f"[{clf_label} | {cohort.upper()}] Gene Backprop",
                )

            if "metrics" in selected_steps:
                ProgressTracker.start_phase(
                    f"{clf_key}_{cohort}_metrics",
                    f"[{clf_label} | {cohort.upper()}] Sensitivity & Specificity..."
                )
                probs_pos = np.zeros(len(patient_ids_sorted), dtype=np.float64)
                for p_idx, pid in enumerate(patient_ids_sorted):
                    if cohort == "oof" and pid in val_pid_to_fold:
                        art = fold_artifacts[val_pid_to_fold[pid]]
                        logit = art["intercept_raw"] + float(np.dot(X_pat_mat[p_idx], art["beta_raw"]))
                        probs_pos[p_idx] = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
                    else:
                        # Media delle probabilità sui 5 fold (coerente con il 5-Fold Ensemble del training)
                        fold_p = [
                            1.0 / (1.0 + np.exp(
                                -np.clip(art["intercept_raw"] + float(np.dot(X_pat_mat[p_idx], art["beta_raw"])), -30,
                                         30)))
                            for art in fold_artifacts
                        ]
                        probs_pos[p_idx] = float(np.mean(fold_p))

                evaluate_and_plot_sensitivity_specificity(
                    y_true=y_pat_arr,
                    y_probs=probs_pos,
                    class_names=patient_class_names,
                    output_dir=output_dir,
                    prefix="mil",
                    title_suffix=f"{clf_label} - {cohort.upper()}",
                )

            if "graph" in selected_steps:
                ProgressTracker.start_phase(f"{clf_key}_{cohort}_graph", f"[{clf_label} | {cohort.upper()}] Gene Saliency & Networks...")
                run_step_graph(
                    output_dir=output_dir,
                    patient_ids_sorted=patient_ids_sorted,
                    graphs_by_patient=graphs_by_patient,
                    y_pat_arr=y_pat_arr,
                    patient_sample_sensitivities=patient_sample_sensitivities,
                    gene_shap_mats=gene_shap_mats,
                    graph_cache=graph_cache,
                    gene_map_dict=gene_map_dict,
                    num_unique_genes=num_unique_genes,
                    feature_names=feature_names,
                    patient_class_names=patient_class_names,
                )

            if "importance" in selected_steps:
                ProgressTracker.start_phase(f"{clf_key}_{cohort}_importance", f"[{clf_label} | {cohort.upper()}] Feature Importance...")
                run_step_importance(
                    output_dir=output_dir,
                    patient_ids_sorted=patient_ids_sorted,
                    X_pat_mat=X_pat_mat,
                    y_pat_arr=y_pat_arr,
                    fold_artifacts=fold_artifacts,
                    val_pid_to_fold=val_pid_to_fold,
                    clinical_cols_ref=clinical_cols_ref,
                    classifier_label=clf_label,
                    cohort=cohort,
                )

            if "shap" in selected_steps:
                ProgressTracker.start_phase(f"{clf_key}_{cohort}_shap", f"[{clf_label} | {cohort.upper()}] SHAP Explanations...")
                run_step_shap(
                    output_dir=output_dir,
                    patient_ids_sorted=patient_ids_sorted,
                    X_pat_mat=X_pat_mat,
                    SHAP_pat_mat=SHAP_pat_mat,
                    gene_shap_mats=gene_shap_mats,
                    gene_val_mats=gene_val_mats,
                    fold_artifacts=fold_artifacts,
                    genomic_cols_ref=genomic_cols_ref,
                    clinical_cols_ref=clinical_cols_ref,
                    target_graphs=target_graphs,
                    graph_cache=graph_cache,
                    gene_map_dict=gene_map_dict,
                    num_unique_genes=num_unique_genes,
                    feature_names=feature_names,
                    classifier_label=clf_label,
                    cohort=cohort,
                )

            if "enrichment" in selected_steps:
                ProgressTracker.start_phase(f"{clf_key}_{cohort}_enrichment", f"[{clf_label} | {cohort.upper()}] Pathway Enrichment...")
                run_step_enrichment(output_dir=output_dir, feature_names=feature_names)

            if "tissue_attn" in selected_steps:
                ProgressTracker.start_phase(f"{clf_key}_{cohort}_tissue_attn", f"[{clf_label} | {cohort.upper()}] Tissue Attention...")
                run_step_tissue_attn(
                    output_dir=output_dir,
                    patient_ids_sorted=patient_ids_sorted,
                    patient_sample_sensitivities=patient_sample_sensitivities,
                    classifier_label=clf_label,
                )

            if "pca" in selected_steps:
                ProgressTracker.start_phase(f"{clf_key}_{cohort}_pca", f"[{clf_label} | {cohort.upper()}] Embedding Latent Space (PCA)...")
                run_step_pca(
                    output_dir=output_dir,
                    patient_ids=patient_ids_sorted,
                    X_pat_mat=X_pat_mat,
                    y_pat_arr=y_pat_arr,
                    patient_sample_sensitivities=patient_sample_sensitivities,
                    fold_artifacts=fold_artifacts,
                    val_pid_to_fold=val_pid_to_fold,
                    label_encoder_patient=label_encoder_patient,
                    classifier_label=clf_label,
                    suffix=cohort,
                )

            summarize_four_insights(mil_models_dir, clf_key, cohort)

    ProgressTracker.finish("LinearMIL Explainability Pipeline completed successfully!")


if __name__ == "__main__":
    main()