import ast
import json
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import to_hex, to_rgb
from pyvis.network import Network
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config_file_tree

logging.getLogger("matplotlib").setLevel(logging.WARNING)

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(config_file_tree.ROOT_DIR, ".numba_cache"))
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# =============================================================================
# STEP EXECUTION ORDER & DEPENDENCY RESOLUTION
# =============================================================================
CANONICAL_STEP_ORDER = {
    "gat": ["metrics", "graph", "importance", "shap", "enrichment", "pca"],
    "mil": ["metrics", "graph", "importance", "shap", "enrichment", "tissue_attn", "pca"],
    "linear_mil": ["metrics", "graph", "importance", "shap", "enrichment", "tissue_attn", "pca"],
}

STEP_PREREQUISITES = {
    "gat": {
        "metrics": [],
        "graph": [],
        "importance": ["metrics"],
        "shap": ["graph"],
        "enrichment": ["graph", "shap"],
        "pca": [],
    },
    "mil": {
        "metrics": [],
        "graph": [],
        "importance": ["metrics"],
        "shap": ["graph"],
        "enrichment": ["graph", "shap"],
        "tissue_attn": [],
        "pca": [],
    },
    "linear_mil": {
        "metrics": [],
        "graph": [],
        "importance": ["metrics"],
        "shap": ["graph"],
        "enrichment": ["graph", "shap"],
        "tissue_attn": [],
        "pca": [],
    },
}

DEFAULT_STEP_ALIASES = {
    "boxplots": "graph",
    "sensitivity": "metrics",
    "specificity": "metrics",
    "roc": "metrics",
}


def parse_multi_choice(raw_values, valid_options, alias_map=None):
    """Parse 'all', single values, or comma/space-separated lists into a deduplicated ordered list."""
    if isinstance(raw_values, str):
        raw_values = [raw_values]

    tokens = []
    for item in raw_values:
        for part in str(item).split(","):
            tok = part.strip().lower()
            if not tok:
                continue
            if alias_map and tok in alias_map:
                tok = alias_map[tok]
            tokens.append(tok)

    if not tokens or "all" in tokens:
        return list(valid_options)

    invalid = [t for t in tokens if t not in valid_options]
    if invalid:
        raise ValueError(f"Invalid option(s) {invalid}. Allowed values: {valid_options + ['all']}")

    seen = set()
    return [t for t in tokens if not (t in seen or seen.add(t))]


def resolve_step_sequence(requested_step, model_type: str, available_steps=None) -> list[str]:
    """Resolve requested step(s) together with all prerequisites in canonical execution order."""
    m_type = model_type.lower().strip()
    canonical = CANONICAL_STEP_ORDER.get(m_type, [])
    prereqs = STEP_PREREQUISITES.get(m_type, {})

    raw_list = [requested_step] if isinstance(requested_step, str) else list(requested_step or [])
    raw_items = []
    for item in raw_list:
        for part in str(item).split(","):
            tok = part.strip().lower()
            if tok:
                raw_items.append(DEFAULT_STEP_ALIASES.get(tok, tok))

    if not raw_items or "all" in raw_items:
        steps = list(canonical)
    else:
        required = set()

        def _collect(step_name: str):
            if step_name in required:
                return
            for dep in prereqs.get(step_name, []):
                _collect(dep)
            required.add(step_name)

        for item in raw_items:
            _collect(item)

        steps = [s for s in canonical if s in required] + [s for s in raw_items if s not in canonical]

    if available_steps is not None:
        avail_set = set(available_steps)
        steps = [s for s in steps if s in avail_set]

    return steps


# =============================================================================
# GLOBAL OUTPUT FILENAME CONSTANTS
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

FN_GAT_SAMPLE_PCA_PNG = "gat_sample_embeddings_pca.png"
FN_MIL_PATIENT_PCA_PNG = "mil_patient_embeddings_pca.png"
FN_MIL_SAMPLE_PCA_PNG = "mil_sample_embeddings_pca.png"
FN_GAT_SAMPLE_PCA_PNG_ALL = "gat_sample_embeddings_pca_all.png"
FN_GAT_SAMPLE_PCA_PNG_TEST = "gat_sample_embeddings_pca_test.png"
FN_MIL_PATIENT_PCA_PNG_ALL = "mil_patient_embeddings_pca_all.png"
FN_MIL_PATIENT_PCA_PNG_TEST = "mil_patient_embeddings_pca_test.png"
FN_MIL_SAMPLE_PCA_PNG_ALL = "mil_sample_embeddings_pca_all.png"
FN_MIL_SAMPLE_PCA_PNG_TEST = "mil_sample_embeddings_pca_test.png"


# =============================================================================
# DYNAMIC FILENAME RESOLVERS
# =============================================================================
def _sanitize_slug(text, replace_slash=False):
    slug = str(text).lower().replace(" ", "_")
    return slug.replace("/", "_") if replace_slash else slug


def _is_gat(prefix):
    return str(prefix).lower() == "gat"


def get_performance_report_filenames(prefix):
    p_upper = str(prefix).upper()
    return [f"report_kfold_test_results_{p_upper}.csv", "report_kfold_test_results.csv", f"report_kfold_test_results_{config_file_tree.TASK}.csv"]


def get_gene_shap_tsv_filename(prefix, feat_name):
    return f"{str(prefix).lower()}_gene_{_sanitize_slug(feat_name)}_shap.tsv"


def get_gene_shap_png_filename(prefix, feat_name):
    return f"{str(prefix).lower()}_gene_{_sanitize_slug(feat_name)}_shap_summary.png"


def get_gene_shap_npz_filename(prefix, feat_name):
    return f"{str(prefix).lower()}_gene_{_sanitize_slug(feat_name)}_shap_matrices.npz"


def get_top5_boxplot_png_filename(prefix, feat_name):
    return f"{str(prefix).lower()}_top5_genes_{_sanitize_slug(feat_name, replace_slash=True)}_boxplot.png"


def get_importance_tsv_filename(prefix):
    return FN_GAT_METADATA_IMPORTANCE_TSV if _is_gat(prefix) else FN_MIL_CLINICAL_IMPORTANCE_TSV


def get_importance_png_filename(prefix):
    return FN_GAT_METADATA_IMPORTANCE_PNG if _is_gat(prefix) else FN_MIL_CLINICAL_IMPORTANCE_PNG


def get_edge_attention_tsv_filename(prefix):
    return FN_GAT_EDGE_ATTENTION if _is_gat(prefix) else FN_MIL_EDGE_ATTENTION


def get_gene_attention_map_tsv_filename(prefix):
    return FN_GAT_GENE_ATTENTION_MAP if _is_gat(prefix) else FN_MIL_GENE_ATTENTION_MAP


def get_all_shap_tsv_filename(prefix):
    return FN_GAT_ALL_SHAP_TSV if _is_gat(prefix) else FN_MIL_ALL_SHAP_TSV


def get_all_shap_png_filename(prefix):
    return FN_GAT_ALL_SHAP_PNG if _is_gat(prefix) else FN_MIL_ALL_SHAP_PNG


def get_heatmap_png_filename(prefix):
    return FN_GAT_HEATMAP_PNG if _is_gat(prefix) else FN_MIL_HEATMAP_PNG


def get_net_interactive_html_filename(prefix):
    return FN_GAT_NET_INTERACTIVE_HTML if _is_gat(prefix) else FN_MIL_NET_INTERACTIVE_HTML


def get_net_static_png_filename(prefix):
    return FN_GAT_NET_STATIC_PNG if _is_gat(prefix) else FN_MIL_NET_STATIC_PNG


# =============================================================================
# PROGRESS TRACKING, PATHS & ID UNWRAPPING HELPERS
# =============================================================================
class ProgressTracker:
    """Manage normalized global progress [0.0, 1.0] across weighted execution phases."""

    _phase_ranges = {"default": (0.0, 1.0)}
    _current_phase = "default"
    _last_emit = 0.0

    @classmethod
    def configure(cls, weighted_phases):
        items = list(weighted_phases.items()) if isinstance(weighted_phases, dict) else [(p, 1.0) for p in (weighted_phases or ["default"])]
        total_w = sum(max(float(w), 0.1) for _, w in items) or 1.0
        cls._phase_ranges = {}
        acc = 0.0
        for name, w in items:
            span = max(float(w), 0.1) / total_w
            cls._phase_ranges[name] = (acc, acc + span)
            acc += span
        cls._current_phase = items[0][0]
        cls._last_emit = 0.0
        cls.emit(0.0, "Initializing...", force=True)

    @classmethod
    def start_phase(cls, phase_name, label=None):
        if phase_name in cls._phase_ranges:
            cls._current_phase = phase_name
        cls.update(0.0, label or f"Running {phase_name}...", force=True)

    @classmethod
    def update(cls, local_fraction, label="", force=False, start_frac=0.0, end_frac=1.0):
        p_start, p_end = cls._phase_ranges.get(cls._current_phase, (0.0, 1.0))
        local_clamped = max(0.0, min(1.0, float(local_fraction)))
        sub_pos = start_frac + (end_frac - start_frac) * local_clamped
        cls.emit(p_start + (p_end - p_start) * sub_pos, label, force=force)

    @classmethod
    def finish(cls, label="Completed!"):
        cls.emit(1.0, label, force=True)

    @classmethod
    def emit(cls, global_pct, label="", force=False):
        now = time.time()
        if not force and (now - cls._last_emit) < 0.08 and global_pct < 0.999:
            return
        cls._last_emit = now
        print(f"[PROGRESS] {max(0.0, min(1.0, float(global_pct))):.4f} | {label}", flush=True)


def tracked_tqdm(iterable, desc="Processing", total=None, start_frac=0.0, end_frac=1.0):
    """Wrap an iterable with tqdm while forwarding normalized progress updates."""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    pbar = tqdm(iterable, desc=desc, total=total, file=sys.stderr)
    if not total or total <= 0:
        yield from pbar
        return

    show_all = total <= 30
    ProgressTracker.update(0.0, f"{desc} (0/{total})", force=True, start_frac=start_frac, end_frac=end_frac)
    for idx, item in enumerate(pbar, start=1):
        yield item
        ProgressTracker.update(idx / total, f"{desc} ({idx}/{total})", force=(show_all or idx == total), start_frac=start_frac, end_frac=end_frac)


def to_rel_path(path_str, root_dir=None):
    """Convert an absolute path to a path relative to the project root."""
    if not path_str:
        return ""
    try:
        return os.path.relpath(path_str, root_dir or config_file_tree.ROOT_DIR)
    except Exception:
        return str(path_str)


def _unwrap_id(val):
    """Unwrap nested lists, tuples, arrays, or scalar tensors into a clean string."""
    while isinstance(val, (list, tuple, np.ndarray)):
        val = val[0] if len(val) > 0 else ""
    if hasattr(val, "item"):
        val = val.item()
    return str(val).strip()


def _unwrap_target(val):
    """Extract a scalar target value from tensors, arrays, or nested sequences."""
    if val is None:
        return "Unknown"
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu()
        return val.flatten()[0].item() if val.numel() > 0 else "Unknown"
    if isinstance(val, (list, tuple, np.ndarray)):
        return _unwrap_target(val[0]) if len(val) > 0 else "Unknown"
    return val.item() if hasattr(val, "item") else val


def decode_class_label(val, label_encoder):
    """Decode a numeric class index into its string label, mapping out-of-range/missing values to 'Unknown'."""
    raw_val = _unwrap_target(val)
    if raw_val == "Unknown" or raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "Unknown"
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        try:
            idx = int(float(raw_val))
            if 0 <= idx < len(label_encoder.classes_):
                return str(label_encoder.classes_[idx])
            return "Unknown"
        except (ValueError, TypeError):
            pass
    s = str(raw_val).strip()
    return "Unknown" if s.lower() in {"nan", "none", "null", "-1", ""} else s


def load_gene_mappings(gene_mapping_file):
    """Load {node_index: (gene_id, gene_symbol)} dictionary from TSV."""
    if not os.path.exists(gene_mapping_file):
        return {}
    df_map = pd.read_csv(gene_mapping_file, sep="\t")
    return {int(r["node_index"]): (str(r["gene_id"]), str(r["gene_symbol"])) for _, r in df_map.iterrows()}


def get_gene_name(gene_idx, gene_map_dict):
    """Return (gene_id, gene_symbol) for a graph node index."""
    return gene_map_dict.get(gene_idx, (f"GENE_{gene_idx}", f"Unknown_{gene_idx}"))


# =============================================================================
# MULTI-COLUMN FEATURE GROUPING (ONE-HOT / MULTI-LABEL BINARIZER)
# =============================================================================
def group_multicol_features(feature_cols):
    """Group multi-column encoded features sharing a common prefix before the last '_'."""
    if not feature_cols:
        return []

    prefix_map = {}
    for idx, col in enumerate(feature_cols):
        prefix = col.rsplit("_", 1)[0] if "_" in col else col
        prefix_map.setdefault(prefix, []).append((idx, col))

    groups = []
    for prefix, items in prefix_map.items():
        idxs = [i for i, _ in items]
        cols = [c for _, c in items]
        if len(idxs) > 1:
            legend = ", ".join(f"{k}={c.rsplit('_', 1)[1]}" for k, c in enumerate(cols))
            display_name = f"{prefix} ({legend})"
        else:
            display_name = cols[0]
        groups.append({"name": display_name, "indices": idxs, "cols": cols})
    return groups


def aggregate_multicol_features_shap(extra_grads, extra_vals, extra_cols):
    """Sum SHAP attributions across grouped encoded columns and derive active category indices."""
    agg_grads, agg_vals, agg_names = [], [], []
    for grp in group_multicol_features(extra_cols):
        idxs = grp["indices"]
        if len(idxs) == 1:
            agg_grads.append(extra_grads[:, idxs[0]])
            agg_vals.append(extra_vals[:, idxs[0]])
        else:
            agg_grads.append(extra_grads[:, idxs].sum(axis=1))
            agg_vals.append(np.argmax(extra_vals[:, idxs], axis=1).astype(np.float32))
        agg_names.append(grp["name"])
    return np.column_stack(agg_grads), np.column_stack(agg_vals), agg_names


# =============================================================================
# GRAPH ATTENTION, SALIENCY, PERMUTATION IMPORTANCE & SHAP
# =============================================================================
def _accumulate_undirected_edges(edge_index, weights, num_unique_genes, edge_sum, edge_counts=None):
    """Accumulate undirected PPI edge attention weights into dictionaries."""
    src_nodes = (edge_index[0] % num_unique_genes).cpu().numpy()
    dst_nodes = (edge_index[1] % num_unique_genes).cpu().numpy()
    for i, w in enumerate(weights):
        u, v = int(src_nodes[i]), int(dst_nodes[i])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        edge_sum[key] = edge_sum.get(key, 0.0) + float(w)
        if edge_counts is not None:
            edge_counts[key] = edge_counts.get(key, 0) + 1


def _save_edge_attention_table(edge_sum, denom_fn, gene_map_dict, out_path):
    """Convert accumulated undirected edge weights into a sorted TSV table."""
    rows = []
    for (u, v), total_att in edge_sum.items():
        gid_a, sym_a = get_gene_name(u, gene_map_dict)
        gid_b, sym_b = get_gene_name(v, gene_map_dict)
        rows.append({
            "gene_id_a": gid_a,
            "gene_symbol_a": sym_a,
            "gene_id_b": gid_b,
            "gene_symbol_b": sym_b,
            "interaction": f"{sym_a} - {sym_b}",
            "attention_weight": float(total_att / denom_fn((u, v))),
        })
    df_edges = pd.DataFrame(rows).sort_values(by="attention_weight", ascending=False)
    df_edges.to_csv(out_path, sep="\t", index=False)
    return df_edges


def extract_gat_edge_attention(model, loader, gene_map_dict, num_unique_genes, output_dir, device):
    """Extract sample-level GAT PPI edge attention weights and save to TSV."""
    logging.info("Extracting GAT PPI edge attention weights...")
    model.eval()
    save_dir = os.path.join(output_dir, "gene_networks")
    os.makedirs(save_dir, exist_ok=True)

    edge_sum, edge_counts = {}, {}
    with torch.no_grad():
        for batch in tracked_tqdm(loader, desc="GAT Edge Attention"):
            batch = batch.to(device)
            node_idx = torch.arange(batch.x.size(0), device=device) % num_unique_genes
            x_in = torch.cat([batch.x, model.gene_embedding(node_idx)], dim=-1)
            _, (e_idx, alpha) = model.conv1(x_in, batch.edge_index, return_attention_weights=True)
            _accumulate_undirected_edges(e_idx, alpha.mean(dim=1).cpu().numpy(), num_unique_genes, edge_sum, edge_counts)

    out_path = os.path.join(save_dir, FN_GAT_EDGE_ATTENTION)
    df_edges = _save_edge_attention_table(edge_sum, lambda k: edge_counts[k], gene_map_dict, out_path)
    logging.info(f"Saved GAT edge attention to {to_rel_path(out_path)}\n")
    return df_edges


def extract_mil_edge_attention(model, loader, gene_map_dict, num_unique_genes, output_dir, device):
    """Extract patient-level MIL attention-weighted PPI edge scores and save to TSV."""
    logging.info("Extracting MIL patient-level edge attention weights...")
    model.eval()
    save_dir = os.path.join(output_dir, "gene_networks")
    os.makedirs(save_dir, exist_ok=True)

    edge_sum, total_bags = {}, 0
    with torch.no_grad():
        for item in tracked_tqdm(loader, desc="MIL Edge Attention"):
            batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
            _, attn_weights = model(batch)
            attn_weights = attn_weights.squeeze()

            enc = model.sample_encoder
            if hasattr(enc, "gene_embedding"):
                node_idx = torch.arange(batch.x.size(0), device=device) % num_unique_genes
                x_in = torch.cat([batch.x, enc.gene_embedding(node_idx)], dim=-1)
            else:
                x_in = batch.x

            _, (e_idx, alpha) = enc.conv1(x_in, batch.edge_index, return_attention_weights=True)
            sample_w = attn_weights[batch.batch[e_idx[1]]] if attn_weights.ndim > 0 else attn_weights
            combined_alpha = (alpha.mean(dim=1) * sample_w).cpu().numpy()
            _accumulate_undirected_edges(e_idx, combined_alpha, num_unique_genes, edge_sum)
            total_bags += 1

    out_path = os.path.join(save_dir, FN_MIL_EDGE_ATTENTION)
    df_edges = _save_edge_attention_table(edge_sum, lambda _: total_bags + 1e-6, gene_map_dict, out_path)
    logging.info(f"Saved MIL edge attention to {to_rel_path(out_path)}\n")
    return df_edges


def extract_gene_saliency_and_attention_map(
    model, loader, gene_map_dict, num_unique_genes, feature_names, output_dir, device, prefix="gat"
):
    """Compute normalized 2D Gene x Feature saliency map via |grad * input|."""
    logging.info(f"Extracting {prefix.upper()} 2D Gene x Feature Attention Map...")
    model.eval()
    save_dir = os.path.join(output_dir, "gene_saliency_attention")
    os.makedirs(save_dir, exist_ok=True)

    num_features = len(feature_names)
    sal_sum = torch.zeros((num_unique_genes, num_features), device=device)
    gene_counts = torch.zeros(num_unique_genes, device=device)

    for item in tracked_tqdm(loader, desc=f"{prefix.upper()} Gene-Feature Attention Map"):
        batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
        batch.x.requires_grad = True
        logits, _ = (
            model(batch.x, batch.edge_index, batch.batch, metadata_x=getattr(batch, "metadata_x", None))
            if prefix == "gat" else model(batch)
        )
        preds = logits.argmax(dim=1)
        model.zero_grad()
        logits[range(len(preds)), preds].sum().backward()

        grad_x_in = (batch.x.grad * batch.x).abs()
        node_idx = torch.arange(batch.x.size(0), device=device) % num_unique_genes
        for f_idx in range(num_features):
            sal_sum[:, f_idx].scatter_add_(0, node_idx, grad_x_in[:, f_idx])
        gene_counts.scatter_add_(0, node_idx, torch.ones(batch.x.size(0), device=device))

    avg_sal = (sal_sum / (gene_counts.unsqueeze(1) + 1e-6)).detach().cpu().numpy()
    norm_sal = avg_sal / (avg_sal.max() if avg_sal.max() > 0 else 1.0)

    rows = []
    for idx in range(num_unique_genes):
        if gene_counts[idx] > 0:
            gid, sym = get_gene_name(idx, gene_map_dict)
            rows.append({
                "gene_id": gid,
                "gene_name": sym,
                "saliency_score": float(norm_sal[idx].mean()),
                **{feat: float(norm_sal[idx, f_i]) for f_i, feat in enumerate(feature_names)},
            })

    df_map = pd.DataFrame(rows).sort_values(by="saliency_score", ascending=False)
    out_path = os.path.join(save_dir, get_gene_attention_map_tsv_filename(prefix))
    df_map.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Saved Gene x Feature attention map to {to_rel_path(out_path)}\n")
    return df_map


def compute_feature_importance(
    model, loader, feature_cols, output_dir, device, is_mil=False, prefix="gat", feature_type_label="metadata", n_repeats=10
):
    """Compute grouped permutation accuracy drop for sample metadata or patient clinical features."""
    logging.info(f"Computing grouped {feature_type_label} feature importance (n_repeats={n_repeats})...")
    save_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, get_importance_tsv_filename(prefix))

    if not feature_cols:
        pd.DataFrame([{"feature": "not used", "importance": "not used"}]).to_csv(out_path, sep="\t", index=False)
        return

    model.eval()
    extra_attr = "clinical_x" if is_mil else "metadata_x"
    extra_tensors = [
        getattr(item[0] if isinstance(item, (list, tuple)) else item, extra_attr).cpu()
        for item in loader
        if getattr(item[0] if isinstance(item, (list, tuple)) else item, extra_attr, None) is not None
    ]
    if not extra_tensors:
        pd.DataFrame([{"feature": "not used", "importance": "not used"}]).to_csv(out_path, sep="\t", index=False)
        return

    all_extra = torch.cat(extra_tensors, dim=0)
    num_samples = all_extra.size(0)

    def _eval_accuracy(override_tensor=None):
        correct, total, offset = 0, 0, 0
        with torch.no_grad():
            for item in loader:
                if not is_mil:
                    batch = item.to(device)
                    orig_x = getattr(batch, "metadata_x", None)
                    bs = batch.num_graphs
                    meta_in = override_tensor[offset : offset + bs].to(device) if (override_tensor is not None and orig_x is not None) else orig_x
                    logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=meta_in)
                    target_y = batch.y
                else:
                    p_batch, target_y = (item[0].to(device), item[1].to(device)) if isinstance(item, (list, tuple)) else (item.to(device), item.patient_y.to(device))
                    bs = target_y.size(0)
                    orig_x = getattr(p_batch, "clinical_x", None)
                    if override_tensor is not None and orig_x is not None:
                        p_batch.clinical_x = override_tensor[offset : offset + bs].to(device)
                    logits, _ = model(p_batch)
                    if orig_x is not None:
                        p_batch.clinical_x = orig_x

                correct += int((logits.argmax(dim=1) == target_y).sum())
                total += bs
                offset += bs
        return correct / total if total > 0 else 0.0

    baseline_acc = _eval_accuracy(None)
    rows = []
    for grp in tracked_tqdm(group_multicol_features(feature_cols), desc=f"Grouped {feature_type_label.capitalize()} Permutation"):
        drops = []
        for _ in range(n_repeats):
            perm_tensor = all_extra.clone()
            perm_tensor[:, grp["indices"]] = all_extra[torch.randperm(num_samples)][:, grp["indices"]]
            drops.append(baseline_acc - _eval_accuracy(perm_tensor))
        rows.append({"feature": grp["name"], "importance": float(np.mean(drops)), "importance_std": float(np.std(drops))})

    df_imp = pd.DataFrame(rows).sort_values(by="importance", ascending=False)
    df_imp.to_csv(out_path, sep="\t", index=False)
    return df_imp


def compute_shap_values(
    model, loader, feature_names, extra_cols, output_dir, device,
    gene_map_dict=None, num_unique_genes=None, is_mil=False, prefix="gat", generate_plots=True, top_n=20
):
    """Compute Gradient x Input SHAP attributions for omic node features and auxiliary covariates."""
    logging.info(f"Computing SHAP values for {prefix.upper()} model...")
    model.eval()
    save_dir = os.path.join(output_dir, "shap_analysis")
    os.makedirs(save_dir, exist_ok=True)

    all_omic_grads, all_omic_vals, all_extra_grads, all_extra_vals = [], [], [], []
    track_genes = gene_map_dict is not None and num_unique_genes is not None
    shap_by_gene = [[[] for _ in feature_names] for _ in range(num_unique_genes)] if track_genes else None
    vals_by_gene = [[[] for _ in feature_names] for _ in range(num_unique_genes)] if track_genes else None

    for item in tracked_tqdm(loader, desc=f"{prefix.upper()} SHAP Gradients", start_frac=0.0, end_frac=0.55):
        if not is_mil:
            batch = item.to(device)
            batch.x.requires_grad = True
            extra_x = getattr(batch, "metadata_x", None)
            if extra_x is not None:
                extra_x.requires_grad = True
            logits, _ = model(batch.x, batch.edge_index, batch.batch, metadata_x=extra_x)
            num_samples = batch.num_graphs
        else:
            batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
            batch.x.requires_grad = True
            extra_x = getattr(batch, "clinical_x", None)
            if extra_x is not None:
                extra_x.requires_grad = True
            logits, _ = model(batch)
            num_samples = logits.size(0)

        # Signed logit difference (>0 favors class 1, <0 favors class 0) for binary classification
        if logits.size(1) == 2:
            target_score = (logits[:, 1] - logits[:, 0]).sum()
        else:
            preds = logits.argmax(dim=1)
            target_score = logits[range(len(preds)), preds].sum()

        model.zero_grad()
        target_score.backward()

        node_attr = (batch.x.grad * batch.x).detach().cpu().numpy()
        node_vals = batch.x.detach().cpu().numpy()
        batch_ids = batch.batch.cpu().numpy()

        sample_attr = np.zeros((num_samples, len(feature_names)))
        sample_vals = np.zeros((num_samples, len(feature_names)))
        for s_idx in range(num_samples):
            mask = batch_ids == s_idx
            if np.any(mask):
                sample_attr[s_idx] = node_attr[mask].mean(axis=0)
                sample_vals[s_idx] = node_vals[mask].mean(axis=0)
        all_omic_grads.append(sample_attr)
        all_omic_vals.append(sample_vals)

        if track_genes:
            node_indices = (torch.arange(batch.x.size(0), device=device) % num_unique_genes).cpu().numpy()
            for idx, g_idx in enumerate(node_indices):
                for f_idx in range(len(feature_names)):
                    shap_by_gene[int(g_idx)][f_idx].append(float(node_attr[idx, f_idx]))
                    vals_by_gene[int(g_idx)][f_idx].append(float(node_vals[idx, f_idx]))

        if extra_x is not None and extra_x.grad is not None:
            all_extra_grads.append((extra_x.grad * extra_x).detach().cpu().numpy())
            all_extra_vals.append(extra_x.detach().cpu().numpy())

    if all_omic_grads:
        comb_grads = np.concatenate(all_omic_grads, axis=0)
        comb_vals = np.concatenate(all_omic_vals, axis=0)
        comb_names = list(feature_names)

        if all_extra_grads and extra_cols:
            agg_g, agg_v, agg_n = aggregate_multicol_features_shap(
                np.concatenate(all_extra_grads, axis=0), np.concatenate(all_extra_vals, axis=0), list(extra_cols)
            )
            comb_grads = np.hstack([comb_grads, agg_g])
            comb_vals = np.hstack([comb_vals, agg_v])
            comb_names += agg_n

        pd.DataFrame({
            "feature": comb_names,
            "shap_importance": np.abs(comb_grads).mean(axis=0),
        }).sort_values(by="shap_importance", ascending=False).to_csv(
            os.path.join(save_dir, get_all_shap_tsv_filename(prefix)), sep="\t", index=False
        )

        if generate_plots:
            ProgressTracker.update(0.62, f"{prefix.upper()} All-Features SHAP Plot...", force=True)
            plot_shap_summary_beeswarm(
                comb_grads, comb_vals, comb_names,
                title=f"{prefix.upper()} All Features SHAP Summary",
                output_path=os.path.join(save_dir, get_all_shap_png_filename(prefix)),
            )

    if track_genes:
        gene_ids, gene_symbols = zip(*(get_gene_name(g_idx, gene_map_dict) for g_idx in range(num_unique_genes)))
        for f_idx, feat_name in tracked_tqdm(list(enumerate(feature_names)), desc=f"{prefix.upper()} SHAP Plots", start_frac=0.65, end_frac=1.0):
            full_shap = np.column_stack([np.array(shap_by_gene[g][f_idx], dtype=np.float32) for g in range(num_unique_genes)])
            full_val = np.column_stack([np.array(vals_by_gene[g][f_idx], dtype=np.float32) for g in range(num_unique_genes)])

            np.savez_compressed(
                os.path.join(save_dir, get_gene_shap_npz_filename(prefix, feat_name)),
                shap_values=full_shap,
                features=full_val,
                gene_names=np.array(gene_symbols, dtype=object),
                gene_ids=np.array(gene_ids, dtype=object),
            )

            df_gene = pd.DataFrame({
                "gene_id": gene_ids,
                "gene_name": gene_symbols,
                "gene_idx": np.arange(num_unique_genes),
                "shap_importance": (np.abs(full_shap).mean(axis=0) if full_shap.size else np.zeros(num_unique_genes)).astype(float),
                "mean_signed_shap": (full_shap.mean(axis=0) if full_shap.size else np.zeros(num_unique_genes)).astype(float),
            }).sort_values(by="shap_importance", ascending=False)

            df_gene.drop(columns=["gene_idx"]).to_csv(
                os.path.join(save_dir, get_gene_shap_tsv_filename(prefix, feat_name)), sep="\t", index=False
            )

            if generate_plots:
                top_df = df_gene.head(top_n)
                sel_idx = top_df["gene_idx"].values
                plot_shap_summary_beeswarm(
                    full_shap[:, sel_idx], full_val[:, sel_idx], top_df["gene_name"].tolist(),
                    title=f"{prefix.upper()} Top {len(sel_idx)} Genes SHAP Summary ({feat_name})",
                    output_path=os.path.join(save_dir, get_gene_shap_png_filename(prefix, feat_name)),
                )


def collect_top_genes_node_features(loader, top_gene_ids, ensg_to_idx, num_unique_genes, is_mil=False):
    """Collect per-sample or patient-averaged node features for selected genes."""
    records = []
    desc = "Collecting Top Genes (MIL)" if is_mil else "Collecting Top Genes (GAT)"
    for item in tracked_tqdm(loader, desc=desc):
        if not is_mil:
            batch_y, batch_x = item.y.cpu().numpy(), item.x.cpu().numpy()
            for g_idx in range(item.num_graphs):
                gx = batch_x[g_idx * num_unique_genes : (g_idx + 1) * num_unique_genes]
                records.append({"label": int(batch_y[g_idx]), **{gid: gx[ensg_to_idx[gid]] if gid in ensg_to_idx else np.nan for gid in top_gene_ids}})
        else:
            p_batch, target_y = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item, item.patient_y)
            px, targets_np = p_batch.x.cpu().numpy(), target_y.cpu().numpy().flatten()
            ns_per_pat = (
                p_batch.num_samples.cpu().numpy().flatten()
                if hasattr(p_batch, "num_samples") and p_batch.num_samples is not None
                else np.array([p_batch.batch.max().item() + 1])
            )
            offset = 0
            for p_idx, n_tissues in enumerate(ns_per_pat):
                n_nodes = int(n_tissues * num_unique_genes)
                mean_x = px[offset : offset + n_nodes].reshape(int(n_tissues), num_unique_genes, -1).mean(axis=0)
                offset += n_nodes
                records.append({"label": int(targets_np[p_idx]), **{gid: mean_x[ensg_to_idx[gid]] if gid in ensg_to_idx else np.nan for gid in top_gene_ids}})
    return records


def run_posthoc_pathway_enrichment(
    saliency_tsv_path, output_dir, top_n_genes=150, shap_tsv_paths=None,
    gsea_gene_sets=None, directional_ora=True,
):
    """
    Execute functional enrichment:
      1) Enrichr ORA (Hallmark, KEGG, Reactome, GO BP) overall and split by direction
         (positive resistance-associated vs. negative sensitivity-associated genes).
      2) Preranked GSEA on signed SHAP scores restricted to active genes.
    """
    import gseapy as gp

    if gsea_gene_sets is None:
        gsea_gene_sets = ["MSigDB_Hallmark_2020"]
    ora_libs = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"]

    noisy_loggers = ["fontTools", "fontTools.subset", "fontTools.ttLib", "gseapy"]
    prev_levels = {name: logging.getLogger(name).level for name in noisy_loggers}
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR if name == "gseapy" else logging.WARNING)

    try:
        enrich_dir = os.path.join(output_dir, "pathway_enrichment")
        os.makedirs(enrich_dir, exist_ok=True)
        df_saliency = pd.read_csv(saliency_tsv_path, sep="\t") if (saliency_tsv_path and os.path.exists(saliency_tsv_path)) else pd.DataFrame()

        valid_shap = [p for p in (shap_tsv_paths or []) if p and os.path.exists(p)]
        if valid_shap:
            per_feat_dfs = []
            for sp in valid_shap:
                df_raw = pd.read_csv(sp, sep="\t")
                if "mean_signed_shap" not in df_raw.columns:
                    df_raw["mean_signed_shap"] = df_raw["shap_importance"]
                df_s = df_raw[["gene_name", "shap_importance", "mean_signed_shap"]].dropna(subset=["gene_name"])
                per_feat_dfs.append(df_s.loc[df_s.groupby("gene_name")["shap_importance"].idxmax()])

            df_rank = pd.concat(per_feat_dfs, ignore_index=True).groupby("gene_name", as_index=False).agg(
                {"shap_importance": "sum", "mean_signed_shap": "sum"}
            )
            ora_col, gsea_col = "shap_importance", "mean_signed_shap"
            logging.info(f"[ENRICHMENT] Using SHAP rankings from {len(valid_shap)} file(s).")
        elif not df_saliency.empty:
            df_rank = df_saliency.copy()
            ora_col = "saliency_score"
            gsea_col = "signed_resistance_score" if "signed_resistance_score" in df_rank.columns else "saliency_score"
            logging.info("[ENRICHMENT] SHAP TSV not found; falling back to saliency_score.")
        else:
            logging.warning("[ENRICHMENT] Neither SHAP nor saliency TSV found; skipping enrichment.")
            return

        df_nonzero = df_rank[df_rank[ora_col] > 0].copy()
        top_genes = df_nonzero.sort_values(by=[ora_col, "gene_name"], ascending=[False, True]).head(top_n_genes)["gene_name"].dropna().tolist()

        ProgressTracker.update(0.10, "Running Enrichr ORA (Overall + Directional)...", force=True)
        df_res = gp.enrichr(gene_list=top_genes, gene_sets=ora_libs, organism="human", outdir=os.path.join(enrich_dir, "enrichr_results")).results
        df_res[df_res["Gene_set"] == "MSigDB_Hallmark_2020"].sort_values(by="Adjusted P-value").to_csv(
            os.path.join(enrich_dir, "hallmark_top_pathways.tsv"), sep="\t", index=False
        )
        kegg_df = df_res[df_res["Gene_set"] == "KEGG_2021_Human"].sort_values(by="Adjusted P-value")
        kegg_df.to_csv(os.path.join(enrich_dir, FN_KEGG_TOP_PATHWAYS_TSV), sep="\t", index=False)

        if not kegg_df.empty:
            gp.barplot(
                kegg_df.head(10), title=f"KEGG Pathway Enrichment (Top {top_n_genes} Genes)",
                top_term=10, figsize=(7, 5), ofname=os.path.join(enrich_dir, FN_KEGG_BARPLOT_PNG),
            )

        if directional_ora and gsea_col in df_nonzero.columns and (df_nonzero[gsea_col] < 0).any():
            for dir_label, sub_df in [
                ("resistance", df_nonzero[df_nonzero[gsea_col] > 0]),
                ("sensitivity", df_nonzero[df_nonzero[gsea_col] < 0]),
            ]:
                dir_genes = sub_df.sort_values(by=[ora_col, "gene_name"], ascending=[False, True]).head(min(100, top_n_genes))["gene_name"].dropna().tolist()
                if len(dir_genes) >= 10:
                    enr_dir = gp.enrichr(gene_list=dir_genes, gene_sets=ora_libs, organism="human", outdir=os.path.join(enrich_dir, f"enrichr_{dir_label}"))
                    enr_dir.results.sort_values(by="Adjusted P-value").to_csv(
                        os.path.join(enrich_dir, f"ora_{dir_label}_pathways.tsv"), sep="\t", index=False
                    )

        rnk = (
            df_nonzero[["gene_name", gsea_col]]
            .dropna()
            .assign(gene_name=lambda d: d["gene_name"].astype(str).str.strip().str.upper())
            .groupby("gene_name", as_index=False)[gsea_col]
            .sum()
            .sort_values(by=[gsea_col, "gene_name"], ascending=[False, True])
            .reset_index(drop=True)
        )
        if rnk[gsea_col].duplicated().any():
            rnk[gsea_col] += np.linspace(1e-7, -1e-7, len(rnk))

        if len(rnk) >= 15:
            ProgressTracker.update(0.65, "Running Preranked GSEA...", force=True)
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=gsea_gene_sets,
                threads=4,
                permutation_num=1000,
                outdir=os.path.join(enrich_dir, "gsea_prerank_results"),
                seed=42,
                verbose=False,
            )
            if pre_res is not None and getattr(pre_res, "res2d", None) is not None and not pre_res.res2d.empty:
                gsea_df = pre_res.res2d.copy()
                sort_col = "FDR q-val" if "FDR q-val" in gsea_df.columns else "NOM p-val"
                if sort_col in gsea_df.columns:
                    gsea_df = gsea_df.sort_values(by=sort_col, ascending=True)
                gsea_df.to_csv(os.path.join(enrich_dir, "gsea_preranked_results.tsv"), sep="\t", index=False)
    finally:
        for name, level in prev_levels.items():
            logging.getLogger(name).setLevel(level)


def _build_metadata_sample_id_lookup():
    """Build a fallback lookup {(patient_id, tissue_type): deque([sample_id, ...])} from metadata_for_training.tsv."""
    from collections import deque

    meta_path = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
    if not os.path.exists(meta_path):
        return {}, {}

    try:
        df_meta = pd.read_csv(meta_path, sep="\t")
    except Exception:
        return {}, {}

    if not {"patient_id", "sample_id"}.issubset(df_meta.columns):
        return {}, {}

    tissue_col = getattr(config_file_tree, "SAMPLE_TARGET_COL", "tissue_type")
    has_tissue = tissue_col in df_meta.columns

    by_pat_tissue, by_pat = {}, {}
    for _, row in df_meta.iterrows():
        pid = str(row["patient_id"]).strip()
        sid = str(row["sample_id"]).strip()
        by_pat.setdefault(pid, deque()).append(sid)
        if has_tissue:
            tname = str(row[tissue_col]).strip()
            by_pat_tissue.setdefault((pid, tname), deque()).append(sid)

    return by_pat_tissue, by_pat


def _extract_batch_sample_ids(p_batch, pid_per_sample, tissue_names, num_weights, meta_lookup_pair):
    """Extract full per-sample IDs from PyG Data/Batch attributes, falling back to metadata_for_training.tsv."""
    raw_sids = next((getattr(p_batch, a) for a in ("sample_id", "sample_ids", "sample_name", "sample_names") if getattr(p_batch, a, None) is not None), None)

    if raw_sids is not None:
        flat_sids = []

        def _flatten(obj):
            if isinstance(obj, (list, tuple, np.ndarray)):
                for item in obj:
                    _flatten(item)
            elif hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
                _flatten(obj.tolist())
            else:
                s = _unwrap_id(obj)
                if s:
                    flat_sids.append(s)

        _flatten(raw_sids)
        if len(flat_sids) == num_weights:
            return flat_sids

    by_pat_tissue, by_pat = meta_lookup_pair
    resolved_sids, per_patient_counter = [], {}
    for pid, tname in zip(pid_per_sample, tissue_names):
        if (pid, tname) in by_pat_tissue and by_pat_tissue[(pid, tname)]:
            sid = by_pat_tissue[(pid, tname)].popleft()
            if pid in by_pat and sid in by_pat[pid]:
                by_pat[pid].remove(sid)
            resolved_sids.append(sid)
        elif pid in by_pat and by_pat[pid]:
            resolved_sids.append(by_pat[pid].popleft())
        else:
            idx = per_patient_counter.get(pid, 0) + 1
            per_patient_counter[pid] = idx
            resolved_sids.append(f"{pid}_sample_{idx}")

    return resolved_sids


def extract_mil_tissue_attention_aggregation(model, loader, output_dir, device, label_encoder_sample=None):
    """Aggregate MIL biopsy attention weights per tissue type across all and multi-tissue bags."""
    logging.info("Extracting MIL attention aggregation by tissue class...")
    model.eval()
    save_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(save_dir, exist_ok=True)

    meta_lookup_pair = _build_metadata_sample_id_lookup()
    records = []

    with torch.no_grad():
        for item in tracked_tqdm(loader, desc="MIL Tissue Attention Aggregation"):
            p_batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)
            _, attn_weights = model(p_batch)
            weights = np.atleast_1d(attn_weights.squeeze().cpu().numpy())

            t_indices = (
                p_batch.y.cpu().numpy()
                if hasattr(p_batch, "y") and p_batch.y is not None
                else np.zeros(len(weights), dtype=int)
            )
            tissue_names = [
                str(label_encoder_sample.classes_[t])
                if (
                    label_encoder_sample is not None
                    and hasattr(label_encoder_sample, "classes_")
                    and t < len(label_encoder_sample.classes_)
                )
                else f"Tissue_{t}"
                for t in t_indices
            ]

            p_ids = getattr(p_batch, "patient_id", "Unknown")
            if hasattr(p_batch, "num_samples") and p_batch.num_samples is not None:
                ns_list = (
                    p_batch.num_samples.cpu().numpy().tolist()
                    if hasattr(p_batch.num_samples, "cpu")
                    else p_batch.num_samples
                )
                pid_per_sample = []
                for p_idx, n_s in enumerate(ns_list):
                    curr_pid = (
                        _unwrap_id(p_ids[p_idx])
                        if isinstance(p_ids, (list, tuple)) and p_idx < len(p_ids)
                        else _unwrap_id(p_ids)
                    )
                    pid_per_sample.extend([curr_pid] * n_s)
            else:
                pid_per_sample = [_unwrap_id(p_ids)] * len(weights)

            sample_ids_batch = _extract_batch_sample_ids(
                p_batch=p_batch,
                pid_per_sample=pid_per_sample,
                tissue_names=tissue_names,
                num_weights=len(weights),
                meta_lookup_pair=meta_lookup_pair,
            )

            for p_id, s_id, t_name, w in zip(pid_per_sample, sample_ids_batch, tissue_names, weights):
                records.append({
                    "patient_id": p_id,
                    "sample_id": s_id,
                    "tissue_type": t_name,
                    "attention_weight": float(w),
                })

    df_rec = pd.DataFrame(records)
    if not df_rec.empty:
        df_rec["patient_id"] = df_rec["patient_id"].astype(str)
        df_rec["sample_id"] = df_rec["sample_id"].astype(str)
        df_rec["tissue_type"] = df_rec["tissue_type"].astype(str)
        p_stats = df_rec.groupby("patient_id").agg(
            bag_size=("tissue_type", "count"),
            num_unique_tissues=("tissue_type", lambda x: len(set(x))),
        ).reset_index()
        df_rec = df_rec.merge(p_stats, on="patient_id", how="left")

        df_agg_all = df_rec.groupby("tissue_type").agg(
            mean_attention_all=("attention_weight", "mean"),
            sample_count_all=("attention_weight", "count"),
        ).reset_index()
        df_mixed = df_rec[df_rec["num_unique_tissues"] > 1]
        if not df_mixed.empty:
            df_agg_mix = df_mixed.groupby("tissue_type").agg(
                mean_attention_mixed=("attention_weight", "mean"),
                std_attention_mixed=("attention_weight", "std"),
                median_attention_mixed=("attention_weight", "median"),
                sample_count_mixed=("attention_weight", "count"),
            ).reset_index()
            df_agg = df_agg_all.merge(df_agg_mix, on="tissue_type", how="left").fillna(0.0)
        else:
            df_agg = df_agg_all.assign(
                mean_attention_mixed=df_agg_all["mean_attention_all"],
                std_attention_mixed=0.0,
                median_attention_mixed=df_agg_all["mean_attention_all"],
                sample_count_mixed=0,
            )

        df_agg["pct_attention_all"] = (df_agg["mean_attention_all"] / max(df_agg["mean_attention_all"].sum(), 1e-8)) * 100.0
        df_agg["pct_attention_mixed"] = (df_agg["mean_attention_mixed"] / max(df_agg["mean_attention_mixed"].sum(), 1e-8)) * 100.0
        df_agg = df_agg.sort_values(by="mean_attention_mixed", ascending=False)
    else:
        df_agg = pd.DataFrame()

    df_agg.to_csv(os.path.join(save_dir, FN_MIL_TISSUE_ATTENTION_AGGREGATION), sep="\t", index=False)
    df_rec.to_csv(os.path.join(save_dir, "mil_patient_tissue_attention_raw.tsv"), sep="\t", index=False)
    return df_agg, df_rec


# =============================================================================
# VISUALIZATION HELPERS (BARPLOTS, BOXPLOTS, NETWORKS, HEATMAPS, BEESWARMS)
# =============================================================================
def plot_importance_barplot(df_importance, x_col, y_col, title, output_path, top_n=20, orientation="horizontal"):
    """Plot horizontal or vertical barplot for feature importances or saliency scores."""
    if df_importance.empty or "not used" in df_importance[x_col].values:
        return
    df_top = df_importance.head(top_n).copy()
    df_top[x_col] = pd.to_numeric(df_top[x_col], errors="coerce")

    plt.figure(figsize=(10, 6))
    if orientation == "vertical":
        sns.barplot(data=df_top, x=y_col, y=x_col, palette="RdPu_r", hue=y_col, legend=False)
        plt.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(y_col.replace("_", " ").capitalize(), fontsize=12)
        plt.ylabel(x_col.replace("_", " ").capitalize(), fontsize=12)
    else:
        sns.barplot(data=df_top, x=x_col, y=y_col, palette="RdPu_r", hue=y_col, legend=False)
        plt.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
        plt.xlabel(x_col.replace("_", " ").capitalize(), fontsize=12)
        plt.ylabel(y_col.replace("_", " ").capitalize(), fontsize=12)

    plt.title(f"{title} (Top {min(top_n, len(df_top))})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gene_feature_boxplots(records, top_genes_dict, class_names, output_dir, prefix="gat", feature_names=None):
    """Plot per-modality boxplots comparing top genes across target classes."""
    if not records:
        return
    sample_val = list(records[0].values())[1]
    n_feats = len(sample_val) if isinstance(sample_val, (np.ndarray, list)) else 1
    if feature_names is None or len(feature_names) != n_feats:
        feature_names = [f"Feature_{i}" for i in range(n_feats)]

    for f_idx, feat_name in enumerate(feature_names):
        df_plot = pd.DataFrame([
            {"label": int(r["label"]), **{gid: float(r[gid][f_idx] if isinstance(r[gid], (np.ndarray, list)) else r[gid]) for gid in top_genes_dict}}
            for r in records
        ])
        unique_lbls = sorted(df_plot["label"].unique())
        lbl_map = {lbl: (class_names[lbl] if lbl < len(class_names) else str(lbl)) for lbl in unique_lbls}

        plt.figure(figsize=(4 * len(top_genes_dict), 6))
        for j, (gid, sym) in enumerate(top_genes_dict.items(), 1):
            plt.subplot(1, len(top_genes_dict), j)
            sns.boxplot(x="label", y=gid, data=df_plot, palette="Spectral", hue="label", legend=False)
            sns.stripplot(x="label", y=gid, data=df_plot, color="black", size=3, alpha=0.5, jitter=0.2)
            plt.title(f"{sym}\n({gid})", fontsize=12, fontweight="bold")
            plt.xticks(ticks=range(len(unique_lbls)), labels=[lbl_map[l] for l in unique_lbls], rotation=15)
            plt.xlabel("Target Class", fontsize=11)
            plt.ylabel(str(feat_name), fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, get_top5_boxplot_png_filename(prefix, feat_name)), dpi=300)
        plt.close()


def plot_interactive_gene_network(tsv_path, output_html, top_n=30, saliency_tsv_path=None):
    """Generate an interactive PyVis PPI network colored by node saliency."""
    if not os.path.exists(tsv_path):
        return
    df = pd.read_csv(tsv_path, sep="\t").head(top_n)
    sal_dict = {}
    if saliency_tsv_path and os.path.exists(saliency_tsv_path):
        df_sal = pd.read_csv(saliency_tsv_path, sep="\t")
        sal_dict = dict(zip(df_sal["gene_name"], df_sal["saliency_score"]))

    G = nx.Graph()
    min_w, max_w = df["attention_weight"].min(), df["attention_weight"].max()
    denom_w = (max_w - min_w) if max_w != min_w else 1.0

    for _, r in df.iterrows():
        u, v, w = str(r["gene_symbol_a"]), str(r["gene_symbol_b"]), float(r["attention_weight"])
        G.add_edge(u, v, weight=w, width=4.0 + 4.0 * ((w - min_w) / denom_w), title=f"{u} - {v}\nAttention: {w:.5f}", color={"color": "rgba(74, 123, 176, 0.85)", "highlight": "#e63946"})

    c_start, c_end = np.array(to_rgb("#B3D9FF")), np.array(to_rgb("#FFB7B2"))
    for node, deg in G.degree():
        sal = sal_dict.get(node, 0.5)
        G.nodes[node].update({
            "size": 25 + 20 * sal,
            "color": {"background": to_hex(c_start * (1.0 - sal) + c_end * sal), "border": "#444444", "highlight": {"background": "#a8dda8", "border": "#222222"}},
            "font": {"color": "#333333", "size": 14, "face": "arial"},
            "title": f"Gene: {node}\nConnections: {deg}\nSaliency: {sal:.5f}",
        })

    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333", directed=False)
    net.from_nx(G)
    net.barnes_hut(gravity=-3000, central_gravity=0.2, spring_length=140)
    net.save_graph(output_html)


def plot_static_gene_network(tsv_path, output_png, top_n=30):
    """Plot a static spring-layout PPI network."""
    if not os.path.exists(tsv_path):
        return
    df = pd.read_csv(tsv_path, sep="\t").head(top_n)
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(str(r["gene_symbol_a"]), str(r["gene_symbol_b"]), weight=float(r["attention_weight"]))
    pos = nx.spring_layout(G, k=0.8, seed=42)

    plt.figure(figsize=(10, 10), facecolor="white")
    nx.draw_networkx_edges(G, pos, edge_color="#888888", alpha=0.75)
    nx.draw_networkx_nodes(G, pos, node_color="#B3D9FF", node_size=700, edgecolors="#444444")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", font_color="#222222")
    plt.title(f"Top {top_n} Gene-Gene PPI Interactions", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_gene_feature_attention_heatmap(tsv_path, output_png, feature_names, top_n=25):
    """Plot a 2D heatmap of normalized gene saliency across omic modalities."""
    if not os.path.exists(tsv_path):
        return
    df_hm = pd.read_csv(tsv_path, sep="\t").head(top_n).set_index("gene_name")[feature_names]
    plt.figure(figsize=(10, max(6, int(top_n * 0.35))))
    sns.heatmap(df_hm, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=0.5)
    plt.title(f"Input Feature Attention Map (Top {top_n} Genes)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_shap_summary_beeswarm(shap_values, features, feature_names, title, output_path, max_display=20):
    """Render a SHAP summary beeswarm plot."""
    if shap_values is None or len(shap_values) == 0:
        return
    try:
        import shap
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features, feature_names=feature_names, max_display=max_display, show=False)
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.warning(f"Failed to generate SHAP beeswarm plot: {e}")
        plt.close()


def generate_filtered_gene_shap_beeswarm(npz_path, target_genes, title, output_path, max_display=30):
    """Render a SHAP beeswarm plot restricted to a user-specified list of target genes."""
    if not npz_path or not os.path.exists(npz_path) or not target_genes:
        return False, 0

    target_set = {str(g).strip().upper() for g in target_genes if str(g).strip()}
    if not target_set:
        return False, 0

    try:
        data = np.load(npz_path, allow_pickle=True)
        shap_mat, feat_mat = data["shap_values"], data["features"]
        gene_names = [str(g) for g in data["gene_names"]]
        gene_ids = [str(g) for g in data["gene_ids"]] if "gene_ids" in data else [""] * len(gene_names)

        matched = [i for i, (sym, gid) in enumerate(zip(gene_names, gene_ids)) if sym.upper() in target_set or gid.upper() in target_set]
        if not matched:
            return False, 0

        order = np.argsort(-np.abs(shap_mat[:, matched]).mean(axis=0))
        sorted_idx = [matched[i] for i in order]
        plot_shap_summary_beeswarm(
            shap_mat[:, sorted_idx], feat_mat[:, sorted_idx], [gene_names[i] for i in sorted_idx],
            title=title, output_path=output_path, max_display=max(max_display, len(sorted_idx)),
        )
        return os.path.exists(output_path), len(sorted_idx)
    except Exception as e:
        logging.warning(f"Failed to generate filtered SHAP beeswarm plot: {e}")
        return False, 0


# =============================================================================
# MODEL TRAINING CONFIGURATION LOADER
# =============================================================================
def load_model_training_config(model_dir):
    """
    Load training configuration associated with a model directory (/GAT, /MIL, /LinearMIL).
    Checks JSON first, then legacy TXT / pretrained_checkpoint_info.txt, falling back to config_file_tree.
    """
    default_cfg = {
        "DATASET_NAME": getattr(config_file_tree, "DATASET_NAME", "N/A"),
        "TASK": getattr(config_file_tree, "TASK", "classification"),
        "GRAPH_EDGES_TYPE": getattr(config_file_tree, "GRAPH_EDGES_TYPE", "N/A"),
        "SAMPLE_TARGET_COL": getattr(config_file_tree, "SAMPLE_TARGET_COL", "tissue_type"),
        "PATIENT_TARGET_COL": getattr(config_file_tree, "PATIENT_TARGET_COL", "is_platinum_resistant"),
        "TIME_COL": getattr(config_file_tree, "TIME_COL", None),
        "NODE_FEATURE_INDEX": list(getattr(config_file_tree, "NODE_FEATURE_INDEX", [0, 1])),
        "NODE_FEATURE_NAMES": list(getattr(config_file_tree, "NODE_FEATURE_NAMES", ["gene_expression", "log2CopyNumber"])),
        "USE_SAMPLES_METADATA": bool(getattr(config_file_tree, "USE_SAMPLES_METADATA", False)),
        "METADATA_FEATURES": list(getattr(config_file_tree, "METADATA_FEATURES", [])),
        "USE_CLINICAL_DATA": bool(getattr(config_file_tree, "USE_CLINICAL_DATA", False)),
        "CLINICAL_FEATURES": list(getattr(config_file_tree, "CLINICAL_FEATURES", [])),
        "SPLIT_DIR_REL": "",
        "PROCESSED_GRAPHS_DIR_REL": "",
        "PROCESSED_SAMPLES_DIR_REL": "",
    }
    if not model_dir:
        return default_cfg

    norm_dir = os.path.normpath(model_dir)
    parent_dir = os.path.dirname(norm_dir)
    sibling_gat_dir = os.path.join(parent_dir, "GAT")

    for candidate in [
        os.path.join(norm_dir, "training_config.json"),
        os.path.join(parent_dir, "training_config.json"),
    ]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                default_cfg.update({k: v for k, v in loaded.items() if v is not None})
                logging.info(f"Loaded JSON configuration: {candidate}")
                return default_cfg
            except Exception:
                pass

    txt_key_map = {
        "Dataset Name": ("DATASET_NAME", str),
        "Task Type": ("TASK", str),
        "Graph Edges Topology": ("GRAPH_EDGES_TYPE", str),
        "Sample Target Label": ("SAMPLE_TARGET_COL", str),
        "Patient Target Label": ("PATIENT_TARGET_COL", str),
        "Active Node Feature Indices": ("NODE_FEATURE_INDEX", ast.literal_eval),
        "Use Samples Metadata": ("USE_SAMPLES_METADATA", ast.literal_eval),
        "Metadata Features Used": ("METADATA_FEATURES", lambda v: [] if v == "None" else ast.literal_eval(v)),
        "Use Patient Clinical Data": ("USE_CLINICAL_DATA", ast.literal_eval),
        "Clinical Features Used": ("CLINICAL_FEATURES", lambda v: [] if v == "None" else ast.literal_eval(v)),
    }
    for candidate in [
        os.path.join(norm_dir, "training_config.txt"),
        os.path.join(parent_dir, "training_config.txt"),
        os.path.join(norm_dir, "pretrained_checkpoint_info.txt"),
        os.path.join(sibling_gat_dir, "pretrained_checkpoint_info.txt"),
    ]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            k, v = (part.strip() for part in line.split(":", 1))
                            if k in txt_key_map and v not in ("N/A", "N/A (MIL Disabled)"):
                                cfg_key, parser_fn = txt_key_map[k]
                                default_cfg[cfg_key] = parser_fn(v)
                logging.info(f"Loaded TXT configuration: {candidate}")
                return default_cfg
            except Exception:
                pass

    logging.info("No training_config file found. Falling back to config_file_tree defaults.")
    return default_cfg


def resolve_experiment_data_dirs(cfg):
    """Resolve (GRAPHS_DIR, SPLIT_DIR, PROCESSED_SAMPLES_DIR) honoring saved config values."""
    root_dir = config_file_tree.ROOT_DIR

    def _replace_in_relpath(base_path, old_token, new_token):
        if not old_token or not new_token or old_token == new_token or new_token == "N/A":
            return base_path
        try:
            rel = os.path.relpath(base_path, root_dir)
            if not rel.startswith("..") and old_token in rel:
                return os.path.join(root_dir, rel.replace(old_token, new_token))
        except ValueError:
            pass
        return base_path.replace(old_token, new_token)

    def _resolve_single(rel_key, fallback_abs):
        rel_val = cfg.get(rel_key, "")
        if rel_val:
            cand = os.path.join(root_dir, rel_val)
            if os.path.exists(cand):
                return cand
        return fallback_abs

    graphs_dir = _resolve_single("PROCESSED_GRAPHS_DIR_REL", config_file_tree.PROCESSED_GRAPHS_DIR)
    split_dir = _resolve_single("SPLIT_DIR_REL", config_file_tree.SPLIT_DIR)
    samples_dir = _resolve_single("PROCESSED_SAMPLES_DIR_REL", config_file_tree.PROCESSED_SAMPLES_DIR)

    saved_dataset = cfg.get("DATASET_NAME", "")
    curr_dataset = getattr(config_file_tree, "DATASET_NAME", "")
    if saved_dataset and saved_dataset != "N/A" and saved_dataset != curr_dataset:
        for label, current_path in [("SPLIT_DIR", split_dir), ("PROCESSED_SAMPLES_DIR", samples_dir), ("GRAPHS_DIR", graphs_dir)]:
            cand = _replace_in_relpath(current_path, curr_dataset, saved_dataset)
            if os.path.exists(cand):
                if label == "SPLIT_DIR":
                    split_dir = cand
                elif label == "PROCESSED_SAMPLES_DIR":
                    samples_dir = cand
                else:
                    graphs_dir = cand

    saved_edges = cfg.get("GRAPH_EDGES_TYPE", "")
    curr_edges = getattr(config_file_tree, "GRAPH_EDGES_TYPE", "")
    if saved_edges and saved_edges != "N/A" and saved_edges != curr_edges:
        parent_graphs = os.path.dirname(os.path.normpath(graphs_dir))
        candidates = [
            _replace_in_relpath(graphs_dir, curr_edges, saved_edges),
            os.path.join(parent_graphs, f"processed_graphs_{saved_edges}"),
            os.path.join(parent_graphs, f"graphs_{saved_edges}"),
            os.path.join(parent_graphs, saved_edges),
        ]
        matched = next((c for c in candidates if c and os.path.exists(c)), None)
        if matched:
            graphs_dir = matched
            logging.info(f"Resolved GRAPHS_DIR for GRAPH_EDGES_TYPE='{saved_edges}': {to_rel_path(graphs_dir)}")
        else:
            logging.warning(
                f"Saved GRAPH_EDGES_TYPE='{saved_edges}' differs from config_file_tree ('{curr_edges}'), "
                f"but no matching folder was found. Falling back to: {to_rel_path(graphs_dir)}"
            )

    return graphs_dir, split_dir, samples_dir


# =============================================================================
# EMBEDDING PROJECTION & CLUSTERING METRICS (PCA / UMAP / SILHOUETTE)
# =============================================================================
PATIENT_TARGET_PALETTE = {
    "yes": "#be185d",
    "no": "#0d9488",
    "Unknown": "#d1d5db",
}
_EXTRA_TARGET_COLORS = ["#d97706", "#7c3aed", "#2563eb", "#059669"]


def _get_target_palette_and_order(series):
    """Build a color mapping (with light grey for 'Unknown') and draw 'Unknown' underneath."""
    unique_vals = [str(v) for v in pd.Series(series).dropna().unique().tolist()]
    known_vals = [v for v in unique_vals if v != "Unknown"]
    ordered_known = [v for v in ("no", "yes") if v in known_vals] + sorted([v for v in known_vals if v not in ("no", "yes")])
    hue_order = (["Unknown"] if "Unknown" in unique_vals else []) + ordered_known

    palette = {}
    extra_idx = 0
    for val in hue_order:
        if val in PATIENT_TARGET_PALETTE:
            palette[val] = PATIENT_TARGET_PALETTE[val]
        else:
            palette[val] = _EXTRA_TARGET_COLORS[extra_idx % len(_EXTRA_TARGET_COLORS)]
            extra_idx += 1
    return palette, hue_order


def _plot_patient_target_scatter(df, x_col, y_col, ax, title, size=55):
    """Plot patient_target points with light grey 'Unknown' in background and vivid 'yes'/'no' on top."""
    palette, hue_order = _get_target_palette_and_order(df["patient_target"])
    for lbl in hue_order:
        sub = df[df["patient_target"] == lbl]
        if sub.empty:
            continue
        is_unk = lbl == "Unknown"
        ax.scatter(
            sub[x_col],
            sub[y_col],
            c=palette[lbl],
            label=lbl,
            s=size * (0.85 if is_unk else 1.05),
            alpha=0.50 if is_unk else 0.90,
            edgecolors="none" if is_unk else "white",
            linewidths=0.45,
            zorder=1 if is_unk else 3,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend(title="patient_target", frameon=True)


def compute_silhouette_only(X_scaled, labels):
    """Compute Silhouette Score on standardized embeddings excluding 'Unknown'/NaN labels."""
    labels_arr = np.asarray(labels, dtype=object)
    valid_mask = pd.notna(labels_arr) & (labels_arr != "Unknown")
    if valid_mask.sum() > 1 and len(np.unique(labels_arr[valid_mask])) > 1:
        try:
            return float(silhouette_score(X_scaled[valid_mask], labels_arr[valid_mask]))
        except Exception:
            pass
    return None


def compute_pca_umap_metrics(embeddings, labels=None):
    """Compute 2D PCA, fast multi-core 2D UMAP, and optional Silhouette Score once per embedding matrix."""
    X_scaled = StandardScaler().fit_transform(embeddings)
    n_samples, n_features = X_scaled.shape

    n_pca = min(32, n_samples - 1, n_features) if (n_samples > 2 and n_features > 2) else min(2, n_features)
    pca = PCA(n_components=max(2, n_pca), random_state=42)
    pca_full = pca.fit_transform(X_scaled)
    pca_res = pca_full[:, :2]
    var_exp = float(pca.explained_variance_ratio_[:2].sum())

    umap_res = None
    if HAS_UMAP and n_samples > 2:
        try:
            umap_input = pca_full if n_features > 32 else X_scaled
            reducer = umap.UMAP(
                n_components=2,
                init="pca",
                n_epochs=150,
                low_memory=False,
                n_jobs=-1,
                n_neighbors=min(15, max(2, n_samples - 1)),
                min_dist=0.1,
            )
            umap_res = reducer.fit_transform(umap_input)
        except Exception as e:
            logging.warning(f"UMAP reduction skipped: {e}")

    score_silh = compute_silhouette_only(X_scaled, labels) if labels is not None else None
    return pca_res, umap_res, var_exp, score_silh, X_scaled


def _plot_sample_embedding_grid(embeddings, df_meta, title_prefix, out_path):
    """Render standardized 1x3 (PCA) or 2x3 (PCA + UMAP) sample embedding diagnostics."""
    pca_res, umap_res, var_exp, score_tissue, X_scaled = compute_pca_umap_metrics(
        embeddings, df_meta["tissue_type"].values
    )
    score_patient = compute_silhouette_only(X_scaled, df_meta["patient_target"].values)

    df_meta["PCA1"], df_meta["PCA2"] = pca_res[:, 0], pca_res[:, 1]
    if umap_res is not None:
        df_meta["UMAP1"], df_meta["UMAP2"] = umap_res[:, 0], umap_res[:, 1]

    df_top = df_meta[df_meta["patient_id"].isin(df_meta["patient_id"].value_counts().head(8).index)]
    n_rows = 2 if umap_res is not None else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    st_str = f" | Silh: {score_tissue:.3f}" if score_tissue is not None else ""
    sp_str = f" | Silh: {score_patient:.3f}" if score_patient is not None else ""

    sns.scatterplot(data=df_meta, x="PCA1", y="PCA2", hue="tissue_type", palette="Set1", alpha=0.85, ax=axes[0, 0])
    axes[0, 0].set_title(f"{title_prefix} PCA by Tissue Type\n(Var Exp: {var_exp:.1%}{st_str})")

    _plot_patient_target_scatter(
        df_meta, "PCA1", "PCA2", axes[0, 1], f"{title_prefix} PCA by Patient Target{sp_str}", size=42
    )

    sns.scatterplot(data=df_top, x="PCA1", y="PCA2", hue="patient_id", palette="tab10", s=90, ax=axes[0, 2])
    axes[0, 2].set_title(f"{title_prefix} Intra-Patient Similarity (Top 8)")

    if umap_res is not None:
        sns.scatterplot(data=df_meta, x="UMAP1", y="UMAP2", hue="tissue_type", palette="Set1", alpha=0.85, ax=axes[1, 0])
        axes[1, 0].set_title(f"{title_prefix} UMAP by Tissue Type")

        _plot_patient_target_scatter(
            df_meta, "UMAP1", "UMAP2", axes[1, 1], f"{title_prefix} UMAP by Patient Target", size=42
        )

        sns.scatterplot(data=df_top, x="UMAP1", y="UMAP2", hue="patient_id", palette="tab10", s=90, ax=axes[1, 2])
        axes[1, 2].set_title(f"{title_prefix} Intra-Patient UMAP")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_gat_pca_plot(
    model, loader, output_dir, device, label_encoder_sample=None, label_encoder_patient=None,
    patient_target_map=None, suffix="all", test_sample_ids=None
):
    """Extract GAT sample embeddings once and save PCA/UMAP plots (for 'all' and optionally 'test')."""
    logging.info(f"Generating GAT sample embeddings PCA plot ({suffix})...")
    model.eval()
    pca_dir = os.path.join(output_dir, "embeddings_pca")
    os.makedirs(pca_dir, exist_ok=True)

    embeddings_list, records = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            meta_x = getattr(batch, "metadata_x", getattr(batch, "metadata_features", None))
            _, emb = model(
                batch.x, batch.edge_index, batch.batch,
                edge_attr=getattr(batch, "edge_attr", None),
                metadata_x=meta_x,
            )
            embeddings_list.append(emb.cpu().numpy())

            p_ids = getattr(batch, "patient_id", "Unknown")
            s_ids = getattr(batch, "sample_id", "Unknown")
            t_labels = getattr(batch, "y", None)
            c_targets = next(
                (getattr(batch, a) for a in ("patient_y", "y_patient", "patient_target", "target", "chemo_refractory_target") if getattr(batch, a, None) is not None),
                None,
            )

            for i in range(emb.size(0)):
                pid = _unwrap_id(p_ids[i]) if isinstance(p_ids, (list, tuple, np.ndarray)) and i < len(p_ids) else _unwrap_id(p_ids)
                sid = _unwrap_id(s_ids[i]) if isinstance(s_ids, (list, tuple, np.ndarray)) and i < len(s_ids) else _unwrap_id(s_ids)
                tissue_name = decode_class_label(t_labels[i] if (t_labels is not None and i < len(t_labels)) else "Unknown", label_encoder_sample)
                raw_c = patient_target_map[pid] if (patient_target_map and pid in patient_target_map) else (c_targets[i] if (c_targets is not None and i < len(c_targets)) else "Unknown")
                records.append({
                    "sample_id": sid,
                    "patient_id": pid,
                    "tissue_type": tissue_name,
                    "patient_target": decode_class_label(raw_c, label_encoder_patient),
                })

    if not embeddings_list:
        logging.warning("No embeddings extracted for GAT PCA plot.")
        return

    all_embs = np.vstack(embeddings_list)
    df_all = pd.DataFrame(records)
    out_path = os.path.join(pca_dir, f"gat_sample_embeddings_pca_{suffix}.png")
    _plot_sample_embedding_grid(all_embs, df_all.copy(), "GAT Sample", out_path)
    logging.info(f"Saved GAT sample PCA plot ({suffix}) to: {out_path}")

    if test_sample_ids is not None:
        test_set = {str(s).strip() for s in test_sample_ids}
        mask = df_all["sample_id"].isin(test_set).values
        if mask.sum() > 2:
            out_test = os.path.join(pca_dir, "gat_sample_embeddings_pca_test.png")
            _plot_sample_embedding_grid(all_embs[mask], df_all[mask].copy().reset_index(drop=True), "GAT Sample", out_test)
            logging.info(f"Saved GAT sample PCA plot (test) to: {out_test}")


def _render_mil_patient_grid(patient_embs, df_p, pca_dir, suffix):
    """Render 1x2 or 2x2 PCA/UMAP grid for MIL patient bag embeddings."""
    pca_res, umap_res, var_exp, score_pt, X_scaled_p = compute_pca_umap_metrics(
        patient_embs, df_p["patient_target"].values
    )
    score_dt = compute_silhouette_only(X_scaled_p, df_p["dominant_tissue"].values)
    df_p["PCA1"], df_p["PCA2"] = pca_res[:, 0], pca_res[:, 1]

    n_rows = 2 if umap_res is not None else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    _plot_patient_target_scatter(
        df_p, "PCA1", "PCA2", axes[0, 0],
        f"Patient Bag Embeddings by Target\n(Var Exp: {var_exp:.1%}{f' | Silh: {score_pt:.3f}' if score_pt is not None else ''})",
        size=95,
    )
    sns.scatterplot(data=df_p, x="PCA1", y="PCA2", hue="dominant_tissue", palette="Set1", s=100, ax=axes[0, 1])
    axes[0, 1].set_title(f"Patient Bag Embeddings by Dominant Tissue{f' | Silh: {score_dt:.3f}' if score_dt is not None else ''}")

    if umap_res is not None:
        df_p["UMAP1"], df_p["UMAP2"] = umap_res[:, 0], umap_res[:, 1]
        _plot_patient_target_scatter(df_p, "UMAP1", "UMAP2", axes[1, 0], "Patient Bag Embeddings UMAP (Target)", size=95)
        sns.scatterplot(data=df_p, x="UMAP1", y="UMAP2", hue="dominant_tissue", palette="Set1", s=100, ax=axes[1, 1])
        axes[1, 1].set_title("Patient Bag Embeddings UMAP (Dominant Tissue)")

    plt.tight_layout()
    out_p_path = os.path.join(pca_dir, f"mil_patient_embeddings_pca_{suffix}.png")
    plt.savefig(out_p_path, dpi=180, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved MIL patient PCA plot ({suffix}) to: {out_p_path}")


def generate_mil_pca_plots(
    model, loader, output_dir, device, label_encoder_sample=None, label_encoder_patient=None,
    patient_target_map=None, suffix="all", test_patient_ids=None
):
    """Extract MIL patient & sample embeddings once and save PCA/UMAP plots ('all' and optionally 'test')."""
    logging.info(f"Generating MIL patient and fine-tuned sample PCA plots ({suffix})...")
    model.eval()
    pca_dir = os.path.join(output_dir, "embeddings_pca")
    os.makedirs(pca_dir, exist_ok=True)

    patient_embs, patient_recs, sample_embs, sample_recs = [], [], [], []
    with torch.no_grad():
        for item in loader:
            p_batch = item[0].to(device) if isinstance(item, (list, tuple)) else item.to(device)

            if hasattr(p_batch, "num_samples") and p_batch.num_samples is not None:
                ns_list = [int(x) for x in np.atleast_1d(p_batch.num_samples.cpu().numpy()).flatten()]
            else:
                ns_list = [int(p_batch.batch.max().item()) + 1] if (hasattr(p_batch, "batch") and p_batch.batch is not None) else [1]
            n_patients = len(ns_list)

            raw_pids = getattr(p_batch, "patient_id", None)
            if isinstance(raw_pids, (list, tuple, np.ndarray)):
                batch_pids = [_unwrap_id(raw_pids[i]) if i < len(raw_pids) else "Unknown" for i in range(n_patients)]
            else:
                single_pid = _unwrap_id(raw_pids) if raw_pids is not None else "Unknown"
                batch_pids = [single_pid] * n_patients

            raw_ptargets = next(
                (getattr(p_batch, attr) for attr in ("patient_y", "y_patient", "patient_target", "target") if getattr(p_batch, attr, None) is not None),
                None,
            )
            if raw_ptargets is None and isinstance(item, (list, tuple)) and len(item) > 1:
                raw_ptargets = item[1]

            if isinstance(raw_ptargets, torch.Tensor):
                ptarget_list = raw_ptargets.detach().cpu().flatten().tolist()
            elif isinstance(raw_ptargets, (list, tuple, np.ndarray)):
                ptarget_list = list(np.atleast_1d(raw_ptargets).flatten())
            elif raw_ptargets is not None:
                ptarget_list = [raw_ptargets] * n_patients
            else:
                ptarget_list = [None] * n_patients

            if hasattr(model, "sample_encoder"):
                meta_x = getattr(p_batch, "metadata_x", getattr(p_batch, "metadata_features", None))
                _, s_emb = model.sample_encoder(
                    p_batch.x, p_batch.edge_index, p_batch.batch,
                    edge_attr=getattr(p_batch, "edge_attr", None),
                    metadata_x=meta_x,
                )
                sample_embs.append(s_emb.cpu().numpy())

                _, attn_weights = model(p_batch)
                attn_1d = attn_weights.reshape(-1)
                t_indices = p_batch.y.cpu().numpy().flatten() if (hasattr(p_batch, "y") and p_batch.y is not None) else np.zeros(s_emb.size(0), dtype=int)
                clin_x = getattr(p_batch, "clinical_x", getattr(p_batch, "clinical_features", None))

                offset = 0
                for k, n_s in enumerate(ns_list):
                    pid = batch_pids[k]
                    if patient_target_map is not None and pid in patient_target_map:
                        raw_t = patient_target_map[pid]
                    elif k < len(ptarget_list) and ptarget_list[k] is not None:
                        raw_t = ptarget_list[k]
                    else:
                        raw_t = "Unknown"
                    p_target_name = decode_class_label(raw_t, label_encoder_patient)

                    bag_tissues = [decode_class_label(t, label_encoder_sample) for t in t_indices[offset : offset + n_s]]
                    for t_name in bag_tissues:
                        sample_recs.append({"patient_id": pid, "tissue_type": t_name, "patient_target": p_target_name})
                    dom_tissue = pd.Series(bag_tissues).mode()[0] if bag_tissues else "Unknown"

                    bag_s_emb = s_emb[offset : offset + n_s]
                    bag_attn = attn_1d[offset : offset + n_s].unsqueeze(-1)
                    pooled_emb = torch.sum(bag_attn * bag_s_emb, dim=0, keepdim=True)
                    if getattr(model, "use_clinical", False) and clin_x is not None:
                        c_row = clin_x[k : k + 1] if clin_x.dim() == 2 else clin_x.unsqueeze(0)
                        pooled_emb = torch.cat([pooled_emb, c_row], dim=-1)

                    patient_embs.append(pooled_emb.cpu().numpy())
                    patient_recs.append({"patient_id": pid, "patient_target": p_target_name, "dominant_tissue": dom_tissue})
                    offset += n_s

    test_pid_set = {str(p).strip() for p in test_patient_ids} if test_patient_ids is not None else None

    if patient_embs:
        all_p_embs = np.vstack(patient_embs)
        df_p_all = pd.DataFrame(patient_recs)
        _render_mil_patient_grid(all_p_embs, df_p_all.copy(), pca_dir, suffix)

        if test_pid_set is not None:
            p_mask = df_p_all["patient_id"].isin(test_pid_set).values
            if p_mask.sum() > 2:
                _render_mil_patient_grid(all_p_embs[p_mask], df_p_all[p_mask].copy().reset_index(drop=True), pca_dir, "test")

    if sample_embs:
        all_s_embs = np.vstack(sample_embs)
        df_s_all = pd.DataFrame(sample_recs)
        out_s_path = os.path.join(pca_dir, f"mil_sample_embeddings_pca_{suffix}.png")
        _plot_sample_embedding_grid(all_s_embs, df_s_all.copy(), "MIL Fine-Tuned Sample", out_s_path)
        logging.info(f"Saved MIL sample PCA plot ({suffix}) to: {out_s_path}")

        if test_pid_set is not None:
            s_mask = df_s_all["patient_id"].isin(test_pid_set).values
            if s_mask.sum() > 2:
                out_s_test = os.path.join(pca_dir, "mil_sample_embeddings_pca_test.png")
                _plot_sample_embedding_grid(all_s_embs[s_mask], df_s_all[s_mask].copy().reset_index(drop=True), "MIL Fine-Tuned Sample", out_s_test)
                logging.info(f"Saved MIL sample PCA plot (test) to: {out_s_test}")


def evaluate_and_plot_sensitivity_specificity(
    y_true,
    y_probs,
    class_names,
    output_dir,
    prefix="mil",
    title_suffix="",
):
    """
    Computes Sensibility (TPR), Specificità (TNR), Balanced Accuracy and optimal Youden threshold,
    saving TSV tables and graphic dashboard (Confusion Matrix + ROC + Trade-off / Per-Class Barplot).
    Support both binary classificators (shape N or Nx2) and multiclass (shape NxC).
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )

    metrics_dir = os.path.join(output_dir, "classification_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64)
    if y_probs.ndim == 1:
        y_probs = np.column_stack([1.0 - y_probs, y_probs])

    n_classes = y_probs.shape[1]
    y_pred = np.argmax(y_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # --- A. Metrics per class (One-vs-Rest) ---
    rows = []
    for c_idx in range(n_classes):
        tp = int(cm[c_idx, c_idx])
        fn = int(cm[c_idx, :].sum() - tp)
        fp = int(cm[:, c_idx].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        bal_acc = 0.5 * (sens + spec)
        y_bin = (y_true == c_idx).astype(int)
        auc_val = float(roc_auc_score(y_bin, y_probs[:, c_idx])) if len(np.unique(y_bin)) > 1 else np.nan

        rows.append({
            "class_idx": c_idx,
            "class_name": str(class_names[c_idx]) if c_idx < len(class_names) else f"Class_{c_idx}",
            "support": tp + fn,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "sensitivity_TPR": sens,
            "specificity_TNR": spec,
            "precision_PPV": ppv,
            "NPV": npv,
            "balanced_accuracy": bal_acc,
            "f1_score": float(f1_score(y_bin, (y_pred == c_idx).astype(int), zero_division=0)),
            "roc_auc": auc_val,
        })

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(metrics_dir, f"{prefix.lower()}_sensitivity_specificity.tsv"), sep="\t", index=False)

    # --- B. Graphic Dashboard (1x3) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    label_suffix = f" ({title_suffix})" if title_suffix else ""

    # Panell 1: Confusion Matrix (counts + % per real class -> diagonal = Sensibility)
    cm_row_pct = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100.0
    annot_mat = np.empty_like(cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot_mat[i, j] = f"{cm[i, j]}\n({cm_row_pct[i, j]:.1f}%)"

    sns.heatmap(
        cm_row_pct,
        annot=annot_mat,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title(f"Confusion Matrix{label_suffix}\n(Row % on diagonal = Sensitivity)")

    if n_classes == 2:
        p_pos = y_probs[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, p_pos)
        auc_bin = roc_auc_score(y_true, p_pos)

        # Standard operating point (threshold 0.5) and optimal (Youden J = TPR - FPR)
        sens_05 = rows[1]["sensitivity_TPR"]
        spec_05 = rows[1]["specificity_TNR"]
        youden_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thresholds[youden_idx])
        sens_opt = float(tpr[youden_idx])
        spec_opt = float(1.0 - fpr[youden_idx])

        # Panel 2: ROC Curve with operating points
        axes[1].plot(fpr, tpr, color="#1d4ed8", lw=2.2, label=f"ROC (AUC = {auc_bin:.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[1].scatter([1 - spec_05], [sens_05], color="#dc2626", s=85, zorder=5,
                        label=f"Thr=0.50 (Sens={sens_05:.2f}, Spec={spec_05:.2f})")
        axes[1].scatter([1 - spec_opt], [sens_opt], color="#059669", marker="D", s=85, zorder=5,
                        label=f"Youden Thr={best_thr:.2f} (Sens={sens_opt:.2f}, Spec={spec_opt:.2f})")
        axes[1].set_xlabel("1 - Specificity (FPR)")
        axes[1].set_ylabel("Sensitivity (TPR)")
        axes[1].set_title(f"ROC & Operating Points{label_suffix}")
        axes[1].legend(loc="lower right", fontsize=8.5)
        axes[1].grid(True, alpha=0.25)

        # Panel 3: Trade-off Sensibility vs Specificity varying threshold
        thr_grid = np.linspace(0.05, 0.95, 91)
        sweep_rows = []
        for t in thr_grid:
            pred_t = (p_pos >= t).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, pred_t, labels=[0, 1]).ravel()
            s_t = tp_t / max(tp_t + fn_t, 1)
            sp_t = tn_t / max(tn_t + fp_t, 1)
            sweep_rows.append({
                "threshold": round(float(t), 3),
                "sensitivity": s_t,
                "specificity": sp_t,
                "balanced_accuracy": 0.5 * (s_t + sp_t),
                "youden_j": s_t + sp_t - 1.0,
            })
        df_sweep = pd.DataFrame(sweep_rows)
        df_sweep.to_csv(os.path.join(metrics_dir, f"{prefix.lower()}_threshold_sweep.tsv"), sep="\t", index=False)

        axes[2].plot(df_sweep["threshold"], df_sweep["sensitivity"], color="#be185d", lw=2.2, label="Sensitivity (TPR)")
        axes[2].plot(df_sweep["threshold"], df_sweep["specificity"], color="#0d9488", lw=2.2, label="Specificity (TNR)")
        axes[2].plot(df_sweep["threshold"], df_sweep["balanced_accuracy"], color="#6b7280", ls=":", lw=1.6, label="Balanced Acc")
        axes[2].axvline(0.5, color="#dc2626", ls="--", lw=1.1, alpha=0.75, label="Default (0.50)")
        axes[2].axvline(best_thr, color="#059669", ls="--", lw=1.1, alpha=0.75, label=f"Youden ({best_thr:.2f})")
        axes[2].set_xlabel("Decision Threshold P(Resistant)")
        axes[2].set_ylabel("Score")
        axes[2].set_ylim(-0.02, 1.05)
        axes[2].set_title(f"Sensitivity vs Specificity Trade-off{label_suffix}")
        axes[2].legend(loc="best", fontsize=8.5)
        axes[2].grid(True, alpha=0.25)
    else:
        # Panel 2 (Multiclass): One-vs-Rest ROC Curves
        for c_idx in range(n_classes):
            y_bin = (y_true == c_idx).astype(int)
            if len(np.unique(y_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_bin, y_probs[:, c_idx])
                axes[1].plot(fpr, tpr, lw=2, label=f"{class_names[c_idx]} (AUC={rows[c_idx]['roc_auc']:.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[1].set_xlabel("1 - Specificity (FPR)")
        axes[1].set_ylabel("Sensitivity (TPR)")
        axes[1].set_title(f"One-vs-Rest ROC{label_suffix}")
        axes[1].legend(loc="lower right", fontsize=8.5)
        axes[1].grid(True, alpha=0.25)

        # Pannello 3 (Multiclass): Barplot Sensibilità vs Specificità per classe
        df_melt = df_metrics.melt(
            id_vars=["class_name"],
            value_vars=["sensitivity_TPR", "specificity_TNR"],
            var_name="Metric",
            value_name="Score",
        )
        df_melt["Metric"] = df_melt["Metric"].map({"sensitivity_TPR": "Sensitivity (TPR)", "specificity_TNR": "Specificity (TNR)"})
        sns.barplot(data=df_melt, x="class_name", y="Score", hue="Metric", palette=["#be185d", "#0d9488"], ax=axes[2])
        axes[2].set_ylim(0, 1.08)
        axes[2].set_xlabel("Class")
        axes[2].set_ylabel("Score")
        axes[2].set_title(f"Per-Class Sensitivity vs Specificity{label_suffix}")
        axes[2].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    out_png = os.path.join(metrics_dir, f"{prefix.lower()}_sensitivity_specificity_dashboard.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved Sensitivity/Specificity dashboard to {to_rel_path(out_png)}")
    return df_metrics