import os
import sys
import ast
import json
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config_file_tree
from utils_explainability import (
    to_rel_path,
    FN_USER_GENE_SUBSETS_JSON,
    FN_KEGG_TOP_PATHWAYS_TSV,
    FN_GAT_SAMPLE_PCA_PNG,
    FN_MIL_PATIENT_PCA_PNG,
    FN_MIL_SAMPLE_PCA_PNG,
    FN_MIL_TISSUE_ATTENTION_AGGREGATION,
    get_performance_report_filenames,
    get_gene_shap_tsv_filename,
    get_gene_shap_png_filename,
    get_gene_shap_npz_filename,
    get_top5_boxplot_png_filename,
    get_importance_tsv_filename,
    get_edge_attention_tsv_filename,
    get_gene_attention_map_tsv_filename,
    get_all_shap_tsv_filename,
    get_all_shap_png_filename,
    get_heatmap_png_filename,
    generate_filtered_gene_shap_beeswarm,
    load_model_training_config,
)

USER_SUBSETS_FILE = os.path.join(config_file_tree.ROOT_DIR, FN_USER_GENE_SUBSETS_JSON)
DELETED_DEFAULTS_KEY = "__deleted_defaults__"
FN_PRETRAINED_GAT_INFO = "pretrained_checkpoint_info.txt"

COHORT_OPTIONS = {
    "Both (PDS + NACT)": "both",
    "NACT": "nact",
    "PDS": "pds",
}

DEFAULT_GENE_GROUPS = {
    "Top Edges Attention | All Genes (Default)": [],
    "Protein Kinases": ["MAPK1", "MAPK3", "AKT1", "PIK3CA", "EGFR", "ERBB2", "CDK1", "CDK2"],
    "DNA Damage & Repair": ["TP53", "BRCA1", "BRCA2", "ATM", "ATR", "PARP1", "CHEK2"],
    "Lipid Metabolism": ["APOC3", "LPL", "FABP4", "ADIPOQ", "PNPLA3", "PPARA"],
    "Cell Adhesion & Angiogenesis": ["CDH5", "RAMP3", "WNT3A", "CALCRL", "COX7A1"],
    "Daria's Paper Relevant Genes": [
        "STAT1", "STAT2", "IRF1", "IRF2", "IRF9", "ISG15", "CXCL10", "CXCL11",
        "CMPK2", "IFIT1", "IFIT3", "OAS1", "CDKN2A", "CDKN2B", "IFNA1", "IFNB1",
        "JAK1", "JAK2", "MX1", "IFIT2", "OAS2", "USP18", "HLA-A", "HLA-B", "HLA-C",
        "B2M", "HIF1A", "VEGFA", "CA9", "BRAF", "CCNE1", "MYC", "KRAS", "NF1",
        "TP53", "BRCA1", "BRCA2", "RAD51C", "RAD51D",
    ],
}

# Embedded HTML/JS search bar overlay injected into the interactive PyVis network
_NETWORK_SEARCH_OVERLAY_HTML = """
<div id="gene-search-overlay" style="
    position: absolute; top: 15px; right: 15px; z-index: 9999;
    background: rgba(255, 255, 255, 0.95); padding: 8px 12px;
    border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    border: 1px solid #cbd5e1; font-family: Arial, sans-serif;
    display: flex; gap: 8px; align-items: center;">
    <input type="text" id="geneSearchInput" placeholder="Search gene symbol..." style="
        padding: 6px 10px; border: 1px solid #94a3b8; border-radius: 6px;
        font-size: 13px; outline: none; width: 170px;"
        onkeypress="if(event.key === 'Enter') performGeneSearch();" />
    <button onclick="performGeneSearch();" style="
        padding: 6px 12px; background-color: #7c3aed; color: white;
        border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;">
        🔍 Search
    </button>
</div>

<div id="geneSearchModal" style="
    display: none; position: absolute; top: 65px; right: 15px; z-index: 10000;
    background-color: #fef2f2; color: #991b1b; border: 1px solid #fca5a5;
    padding: 10px 14px; border-radius: 8px; font-family: Arial, sans-serif;
    font-size: 13px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); align-items: center; gap: 8px;">
    <span id="geneSearchModalText">Gene not found!</span>
    <button onclick="document.getElementById('geneSearchModal').style.display='none';" style="
        background: none; border: none; color: #991b1b; font-weight: bold;
        cursor: pointer; font-size: 14px; margin-left: 6px;">✕</button>
</div>

<script type="text/javascript">
function performGeneSearch() {
    var inputElem = document.getElementById('geneSearchInput');
    var searchVal = inputElem ? inputElem.value.trim().toUpperCase() : "";
    var modal = document.getElementById('geneSearchModal');
    var modalText = document.getElementById('geneSearchModalText');
    if (!searchVal || typeof nodes === 'undefined' || typeof network === 'undefined') return;

    var allNodes = nodes.get();
    var foundNode = allNodes.find(function(n) {
        return String(n.id).toUpperCase() === searchVal || (n.label && String(n.label).toUpperCase() === searchVal);
    }) || allNodes.find(function(n) {
        return String(n.id).toUpperCase().includes(searchVal);
    });

    if (foundNode) {
        if (modal) modal.style.display = 'none';
        network.focus(foundNode.id, { scale: 1.6, animation: { duration: 1000, easingFunction: 'easeInOutQuad' } });
        network.selectNodes([foundNode.id]);
    } else if (modal && modalText) {
        modalText.innerText = "⚠️ Gene '" + searchVal + "' is not present in the currently displayed network graph.";
        modal.style.display = 'flex';
        setTimeout(function() { modal.style.display = 'none'; }, 4500);
    }
}
</script>
"""


# =============================================================================
# PAGE CONFIGURATION & GLOBAL VIOLET THEME STYLING
# =============================================================================
st.set_page_config(
    page_title="GNN & MIL BioExplainability Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    :root {
        --primary-color: #7c3aed;
    }
    .stMarkdown p, .stMarkdown li, div[data-testid="stMarkdownContainer"] p { font-size: 1.22rem !important; line-height: 1.65 !important; }
    .stAlert div[role="alert"] p, div[data-testid="stAlert"] p { font-size: 1.12rem !important; }
    .stCaption, div[data-testid="stCaptionContainer"] p, caption { font-size: 1.05rem !important; }
    label, .stSelectbox label, .stTextInput label, .stRadio label, .stSlider label { font-size: 1.10rem !important; }
    .stDataFrame, div[data-testid="stTable"] { font-size: 1.06rem !important; }

    div[data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] {
        gap: 1.45rem !important;
    }
    div[data-baseweb="tab-panel"] hr {
        margin: 1.8rem 0 !important;
    }
    div[data-baseweb="tab-panel"] h2,
    div[data-baseweb="tab-panel"] h3 {
        margin-top: 0.6rem !important;
        margin-bottom: 0.35rem !important;
    }
    div[data-baseweb="tab-panel"] div[data-testid="stImage"],
    div[data-baseweb="tab-panel"] div[data-testid="stDataFrame"],
    div[data-baseweb="tab-panel"] div[data-testid="stExpander"] {
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #f8f5ff;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #ddd6fe;
    }
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label {
        font-size: 1.05rem !important; font-weight: 600 !important; color: #4c1d95 !important;
    }
    div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] p {
        font-size: 2.1rem !important; font-weight: 700 !important; color: #1e1b4b !important;
    }

    /* Primary buttons (rich violet) */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #7c3aed !important;
        border: 1px solid #7c3aed !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #6d28d9 !important;
        border-color: #6d28d9 !important;
        color: #ffffff !important;
    }

    /* Secondary & download buttons */
    button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"],
    div[data-testid="stDownloadButton"] button {
        background-color: #f3e8ff !important;
        border: 1px solid #c084fc !important;
        color: #4c1d95 !important;
        font-weight: 500 !important;
    }
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #e9d5ff !important;
        border-color: #9333ea !important;
        color: #3b0764 !important;
    }

    /* Selectboxes, multiselects & dropdown popovers */
    div[data-baseweb="select"] > div {
        background-color: #ede9fe !important;
        border: 1.5px solid #8b5cf6 !important;
        color: #1e1b4b !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within {
        background-color: #ddd6fe !important;
        border-color: #7c3aed !important;
    }
    div[data-baseweb="popover"] { max-width: 90vw !important; }
    div[data-baseweb="popover"] [role="listbox"] {
        min-width: max-content !important; max-width: 85vw !important; white-space: nowrap !important;
        background-color: #faf5ff !important;
        border: 1px solid #c4b5fd !important;
    }
    div[data-baseweb="popover"] [role="option"] { white-space: nowrap !important; }
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [aria-selected="true"] {
        background-color: #ede9fe !important;
        color: #4c1d95 !important;
    }
    span[data-baseweb="tag"] {
        background-color: #ddd6fe !important;
        color: #4c1d95 !important;
    }

    /* Clickable expanders */
    div[data-testid="stExpander"] > details {
        border: 1px solid #c4b5fd !important;
        border-radius: 8px !important;
        background-color: #faf5ff !important;
    }
    div[data-testid="stExpander"] > details > summary {
        background-color: #f3e8ff !important;
        color: #4c1d95 !important;
        font-weight: 600 !important;
        border-radius: 7px !important;
    }
    div[data-testid="stExpander"] > details > summary:hover {
        background-color: #e9d5ff !important;
    }

    /* Radio buttons & checkboxes */
    div[data-baseweb="radio"] input:checked + div,
    div[data-baseweb="radio"] > div:first-child > div,
    div[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
    }
    div[data-baseweb="radio"] > div:first-child,
    div[data-baseweb="radio"]:not(:has(input:checked)) > div:first-child {
        background-color: #f5f3ff !important;
        border-color: #a78bfa !important;
    }
    div[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
    }
    div[data-baseweb="checkbox"]:not(:has(input:checked)) > div:first-child {
        background-color: #f5f3ff !important;
        border-color: #a78bfa !important;
    }

    /* Sliders */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
    }
    div[data-baseweb="slider"] > div > div > div:first-child {
        background-color: #7c3aed !important;
    }

    /* Navigation tabs */
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #7c3aed !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #7c3aed !important;
    }
    button[data-baseweb="tab"]:hover,
    button[data-baseweb="tab"]:hover p {
        color: #7c3aed !important;
    }

    /* Uniform inline code sizing */
    code,
    .stMarkdown code,
    div[data-testid="stMarkdownContainer"] code,
    div[data-testid="stAlert"] code {
        font-size: 1.10rem !important;
        padding: 2px 6px !important;
        border-radius: 5px !important;
    }

    /* Text & number inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #ede9fe !important;
        border: 1.5px solid #8b5cf6 !important;
        color: #1e1b4b !important;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        background-color: #f5f3ff !important;
        border-color: #7c3aed !important;
    }

    /* Glassmorphic Overlay Modal */
    div[data-baseweb="modal"] {
        background-color: rgba(30, 27, 75, 0.55) !important;
        backdrop-filter: blur(4px) !important;
    }
    div[data-testid="stDialog"] div[role="dialog"] {
        background-color: rgba(255, 255, 255, 0.94) !important;
        backdrop-filter: blur(10px) !important;
        border: 1.5px solid #c4b5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 20px 50px rgba(30, 27, 75, 0.30) !important;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# UI & FILE HELPERS
# =============================================================================
def short_path(filepath, max_parts=3):
    """Return the last `max_parts` segments of a path for compact UI labels."""
    norm_path = os.path.normpath(to_rel_path(filepath))
    parts = norm_path.split(os.sep)
    return "/".join(parts[-max_parts:]) if len(parts) >= max_parts else norm_path


def path_badge(filepath, mode="rel"):
    """Render a compact monospace path badge with full path details on hover."""
    if not filepath:
        return ""
    rel_path = to_rel_path(filepath)
    if mode == "name":
        label, hover = os.path.basename(filepath), rel_path
        style = "background-color: rgba(124, 58, 237, 0.12); color: #3b0764;"
    elif mode == "short":
        label, hover = short_path(filepath), rel_path
        style = "background-color: rgba(124, 58, 237, 0.12); color: #3b0764;"
    elif mode == "short2":
        label, hover = short_path(filepath, max_parts=2), rel_path
        style = "background-color: rgba(124, 58, 237, 0.12); color: #3b0764;"
    else:
        label, hover = rel_path, os.path.abspath(filepath)
        style = "background-color: rgba(124, 58, 237, 0.10); font-family: monospace;"

    return f'<span title="{hover}" style="cursor: help; {style} padding: 2px 6px; border-radius: 4px;">{label}</span>'


def show_path_alert(alert_type, message_before, filepath, message_after=""):
    """Display a styled alert banner showing the relative file path (with full path on hover)."""
    palette = {
        "warning": ("#f5f3ff", "#c4b5fd", "#4c1d95"),
        "error":   ("#fef2f2", "#fca5a5", "#991b1b"),
        "info":    ("#f5f3ff", "#c4b5fd", "#4c1d95"),
        "success": ("#f0fdf4", "#bbf7d0", "#166534"),
    }
    bg, border, text = palette.get(alert_type, palette["info"])
    full_html = f"{message_before} {path_badge(filepath, mode='rel')} {message_after}".strip()
    st.markdown(
        f'<div style="background-color: {bg}; border: 1px solid {border}; color: {text}; '
        f'padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; font-size: 1.05rem;">{full_html}</div>',
        unsafe_allow_html=True,
    )


def load_user_subsets():
    """Load custom user-defined gene pathways and deleted-default metadata from JSON."""
    if os.path.exists(USER_SUBSETS_FILE):
        try:
            with open(USER_SUBSETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_user_subsets(subsets_dict):
    """Save custom user-defined gene pathways and deleted-default metadata to JSON."""
    try:
        with open(USER_SUBSETS_FILE, "w", encoding="utf-8") as f:
            json.dump(subsets_dict, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving gene subsets: {e}")
        return False


def get_available_gene_groups():
    """Return active pathways combining defaults (minus any deleted ones) and user-saved pathways."""
    raw_data = load_user_subsets()
    deleted_defaults = set(raw_data.get(DELETED_DEFAULTS_KEY, []))
    custom_groups = {k: v for k, v in raw_data.items() if k != DELETED_DEFAULTS_KEY}
    combined = {k: v for k, v in DEFAULT_GENE_GROUPS.items() if k not in deleted_defaults}
    combined.update(custom_groups)
    return combined


def load_tsv(filepath):
    """Safely load a TSV or CSV table into a DataFrame with automatic delimiter detection."""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","
        return pd.read_csv(filepath, sep=sep)
    except Exception:
        try:
            return pd.read_csv(filepath, sep=None, engine="python")
        except Exception:
            return None


def get_explain_file(explain_dir, subfolder, filename):
    """Resolve an explainability artifact inside its subfolder, falling back to root explain_dir."""
    sub_path = os.path.join(explain_dir, subfolder, filename)
    if os.path.exists(sub_path):
        return sub_path
    root_path = os.path.join(explain_dir, filename)
    return root_path if os.path.exists(root_path) else sub_path


def find_experiment_dirs(cohort_key):
    """
    Return [(exp_name, mil_pt_path, linear_mil_dir, gat_pt_path)] matching the active subcohort.
    Each model type (MIL, LinearMIL, GAT) is detected independently.
    """
    base_dir = config_file_tree.TRAINED_MODELS_DIR
    if not os.path.exists(base_dir):
        return []

    expected_tag = None if cohort_key == "both" else cohort_key
    experiments = []
    for exp_name in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_dir) or "mil" not in exp_name.lower():
            continue

        name_lower = exp_name.lower()
        tag = (
            "nact" if ("nact" in name_lower and "pds" not in name_lower)
            else "pds" if ("pds" in name_lower and "nact" not in name_lower)
            else None
        )
        if tag != expected_tag:
            continue

        mil_dir = os.path.join(exp_dir, "MIL")
        mil_lw_dir = os.path.join(exp_dir, "LinearMIL")
        gat_dir = os.path.join(exp_dir, "GAT")

        mil_pt = (
            next((os.path.join(mil_dir, f) for f in sorted(os.listdir(mil_dir)) if f.endswith(".pt")), None)
            if os.path.isdir(mil_dir) else None
        )

        linear_mil_target = None
        if os.path.isdir(mil_lw_dir):
            fold_models_dir = os.path.join(mil_lw_dir, "fold_models")
            has_artifacts = (
                os.path.isdir(fold_models_dir)
                and any(f.endswith(".joblib") for f in os.listdir(fold_models_dir))
            )
            if has_artifacts:
                linear_mil_target = mil_lw_dir

        gat_pt = (
            next((os.path.join(gat_dir, f) for f in sorted(os.listdir(gat_dir)) if f.endswith(".pt")), None)
            if os.path.isdir(gat_dir) else None
        )

        if mil_pt or linear_mil_target or gat_pt:
            experiments.append((exp_name, mil_pt, linear_mil_target, gat_pt))

    # Priorità: 0 = esperimenti con MIL, 1 = esperimenti solo LinearMIL, 2 = solo GAT
    experiments.sort(
        key=lambda item: (
            0 if (item[1] is not None and "linearmil" not in item[0].lower()) else (
                1 if item[2] is not None else 2),
            item[0].lower(),
        )
    )

    return experiments


def resolve_original_gat_checkpoint(gat_pt_path):
    """
    If `gat_pt_path` resides in a cloned `/GAT/` subfolder containing `pretrained_checkpoint_info.txt`,
    resolve and return `(original_gat_pt_path, True)` so sample-level explainability artifacts are
    read/generated directly in the original pre-trained GAT experiment directory.
    Falls back to `(gat_pt_path, False)` if the notice file is absent or the original path is missing.
    """
    if not gat_pt_path:
        return gat_pt_path, False

    local_gat_dir = os.path.dirname(gat_pt_path)
    info_file = os.path.join(local_gat_dir, FN_PRETRAINED_GAT_INFO)
    if not os.path.isfile(info_file):
        return gat_pt_path, False

    orig_abs_dir = None
    orig_rel_dir = None
    pretrained_sub_dir = None

    try:
        with open(info_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line.startswith("Original Pretrained GAT Path:"):
                    orig_abs_dir = line.split(":", 1)[1].strip()
                elif line.startswith("Original Relative Path:"):
                    orig_rel_dir = line.split(":", 1)[1].strip()
                elif line.startswith("Pretrained Sample Encoder Folder (GAT):"):
                    pretrained_sub_dir = line.split(":", 1)[1].strip()
    except Exception:
        return gat_pt_path, False

    candidate_dirs = []
    if orig_abs_dir:
        candidate_dirs.append(os.path.normpath(orig_abs_dir))
    if orig_rel_dir:
        candidate_dirs.append(os.path.normpath(os.path.join(config_file_tree.ROOT_DIR, orig_rel_dir)))
    if pretrained_sub_dir:
        candidate_dirs.append(os.path.normpath(os.path.join(config_file_tree.TRAINED_MODELS_DIR, pretrained_sub_dir)))

    expected_pt_name = os.path.basename(gat_pt_path)
    for cand_dir in candidate_dirs:
        if not cand_dir or not os.path.isdir(cand_dir):
            continue
        if os.path.abspath(cand_dir) == os.path.abspath(local_gat_dir):
            continue

        exact_pt = os.path.join(cand_dir, expected_pt_name)
        if os.path.isfile(exact_pt):
            return exact_pt, True

        any_pt = next(
            (os.path.join(cand_dir, fn) for fn in sorted(os.listdir(cand_dir)) if fn.endswith(".pt")),
            None,
        )
        if any_pt and os.path.isfile(any_pt):
            return any_pt, True

    return gat_pt_path, False


def get_model_config(selected_model_name, selected_exp_name=None):
    """Resolve checkpoint/artifact paths, output folders, and saved training metadata for MIL, LinearMIL, or GAT."""
    is_gat = "GAT" in selected_model_name
    is_lightweight = (selected_model_name == "Patient Level (LinearMIL)")
    level_subdir = "GAT" if is_gat else ("LinearMIL" if is_lightweight else "MIL")

    if is_gat:
        custom_path, _ = resolve_original_gat_checkpoint(st.session_state.get("custom_gat_path"))
        target_key, extra_key, extra_type, prefix = "SAMPLE_TARGET_COL", "USE_SAMPLES_METADATA", "Samples Metadata", "gat"
    elif is_lightweight:
        custom_path = st.session_state.get("custom_linear_mil_path")
        target_key, extra_key, extra_type, prefix = "PATIENT_TARGET_COL", "USE_CLINICAL_DATA", "Patient Clinical Features", "mil"
    else:
        custom_path = st.session_state.get("custom_mil_path")
        target_key, extra_key, extra_type, prefix = "PATIENT_TARGET_COL", "USE_CLINICAL_DATA", "Patient Clinical Features", "mil"

    if custom_path and os.path.exists(custom_path):
        if is_lightweight:
            model_dir = custom_path if os.path.isdir(custom_path) else os.path.dirname(custom_path)
            fold_dir = os.path.join(model_dir, "fold_models")
            model_path = fold_dir if os.path.isdir(fold_dir) else model_dir
        else:
            model_path = custom_path
            model_dir = os.path.dirname(model_path)
        model_exists = True
    elif selected_exp_name:
        model_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, selected_exp_name, level_subdir)
        model_path = model_dir
        model_exists = False
    else:
        model_dir = config_file_tree.TRAINED_MODELS_DIR
        model_path = config_file_tree.TRAINED_MODELS_DIR
        model_exists = False

    saved_cfg = {}
    if model_exists:
        cfg_search_dir = (
            model_dir
            if os.path.exists(os.path.join(model_dir, "training_config.json"))
            else os.path.dirname(model_dir)
        )
        saved_cfg = load_model_training_config(cfg_search_dir)

    use_extra_val = saved_cfg.get(extra_key, getattr(config_file_tree, extra_key))
    if is_lightweight and selected_exp_name:
        if "withClin" in selected_exp_name:
            use_extra_val = True
        elif "noClin" in selected_exp_name:
            use_extra_val = False

    if is_lightweight:
        clf_key = st.session_state.get("lw_classifier", "elasticnet")
        clf_subdir = "PLS-DA" if clf_key == "plsda" else "ElasticNet"
        eval_subset = st.session_state.get("lw_eval_cohort", "oof")
        explain_dir = os.path.join(model_dir, "explained", clf_subdir, eval_subset)
    else:
        explain_dir = os.path.join(model_dir, "explained")
        if model_exists:
            os.makedirs(explain_dir, exist_ok=True)

    return {
        "model_path": model_path,
        "model_dir": model_dir,
        "model_exists": model_exists,
        "is_lightweight": is_lightweight,
        "explain_dir": explain_dir,
        "target_col": saved_cfg.get(target_key, getattr(config_file_tree, target_key)),
        "use_extra": use_extra_val,
        "extra_type": extra_type,
        "prefix": prefix,
        "task": saved_cfg.get("TASK", config_file_tree.TASK),
        "node_feature_indices": saved_cfg.get("NODE_FEATURE_INDEX", getattr(config_file_tree, "NODE_FEATURE_INDEX", [0, 1])),
        "node_feature_names": saved_cfg.get("NODE_FEATURE_NAMES", getattr(config_file_tree, "NODE_FEATURE_NAMES", ["gene_expression", "log2CopyNumber"])),
    }


def load_performance_report(model_dir, prefix, is_lightweight=False, classifier="elasticnet"):
    """Locate and load the K-fold cross-validation metrics CSV/TSV for the selected model."""
    candidates = []
    if is_lightweight and prefix == "mil":
        primary_csv = "report_kfold_test_PLSDA.csv" if classifier == "plsda" else "report_kfold_test_ElasticNet.csv"
        secondary_csv = "report_kfold_test_ElasticNet.csv" if classifier == "plsda" else "report_kfold_test_PLSDA.csv"
        candidates.extend([primary_csv, secondary_csv])
    candidates.extend(get_performance_report_filenames(prefix))

    for fname in candidates:
        report_path = os.path.join(model_dir, fname)
        if os.path.exists(report_path):
            df = load_tsv(report_path)
            if df is not None:
                return df, report_path
    return None, os.path.join(model_dir, candidates[0])


def parse_report_metrics(df):
    """Filter individual CV fold rows and compute mean ± variance across folds."""
    if df is None or df.empty:
        return None, {}

    if "fold" in df.columns:
        summary_labels = {"mean", "std", "var", "variance", "mean ± var", "mean +- var"}
        df_folds = df[~df["fold"].astype(str).str.strip().str.lower().isin(summary_labels)].copy()
    else:
        df_folds = df.copy()

    metrics_summary = {}
    for col in [c for c in df.columns if c != "fold"]:
        vals = pd.to_numeric(df_folds[col], errors="coerce").dropna()
        if not vals.empty:
            mean_val = float(vals.mean())
            var_val = float(vals.var()) if len(vals) > 1 else 0.0
            metrics_summary[col] = {
                "mean": mean_val,
                "var": var_val,
                "display": f"{mean_val:.4f} ± {var_val:.4f}",
            }
    return df_folds, metrics_summary


# =============================================================================
# PATHWAY ENRICHMENT HELPERS
# =============================================================================
ENRICHMENT_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
    "All Libraries",
]
ENRICHMENT_FDR_CUTOFF = 0.05


def load_overall_enrichment_df(explain_dir):
    """Load and merge overall Enrichr ORA results across all queried gene-set libraries."""
    enrich_dir = os.path.join(explain_dir, "pathway_enrichment")
    dfs = []

    for fname in ("hallmark_top_pathways.tsv", FN_KEGG_TOP_PATHWAYS_TSV):
        fpath = get_explain_file(explain_dir, "pathway_enrichment", fname)
        df_part = load_tsv(fpath)
        if df_part is not None and not df_part.empty:
            dfs.append(df_part)

    enrichr_sub = os.path.join(enrich_dir, "enrichr_results")
    if os.path.isdir(enrichr_sub):
        for fname in sorted(os.listdir(enrichr_sub)):
            if fname.endswith(".enrichr.reports.txt") or fname.endswith(".tsv"):
                df_part = load_tsv(os.path.join(enrichr_sub, fname))
                if df_part is not None and not df_part.empty:
                    dfs.append(df_part)

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    if {"Gene_set", "Term"}.issubset(df_all.columns):
        df_all = df_all.drop_duplicates(subset=["Gene_set", "Term"], keep="first")
    if "Adjusted P-value" in df_all.columns:
        df_all["Adjusted P-value"] = pd.to_numeric(df_all["Adjusted P-value"], errors="coerce")
        df_all = df_all.sort_values(by="Adjusted P-value", ascending=True)
    return df_all.reset_index(drop=True)


def filter_enrichment_by_library(df, selected_library):
    """Filter an enrichment DataFrame by the selected gene-set library and sort by Adjusted P-value."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.drop(columns=[c for c in df.columns if "old" in c.lower()]).copy()
    if selected_library != "All Libraries" and "Gene_set" in out.columns:
        out = out[out["Gene_set"] == selected_library].copy()

    if "Adjusted P-value" in out.columns:
        out["Adjusted P-value"] = pd.to_numeric(out["Adjusted P-value"], errors="coerce")
        out = out.sort_values(by="Adjusted P-value", ascending=True)

    return out.reset_index(drop=True)


def render_enrichment_panel(df_lib, title, bar_color, download_filename, fdr_cutoff=ENRICHMENT_FDR_CUTOFF, top_n=8):
    """Display statistically significant pathways (Adjusted P-value < fdr_cutoff) and provide full TSV export."""
    if df_lib is None or df_lib.empty:
        st.info("No enrichment results available for the selected library.")
        return

    df_sig = (
        df_lib[df_lib["Adjusted P-value"] < fdr_cutoff].copy()
        if "Adjusted P-value" in df_lib.columns
        else df_lib.copy()
    )

    if df_sig.empty:
        st.info(
            f"No pathways reached statistical significance (Adjusted P-value < {fdr_cutoff}). "
            "Download the full TSV file below to inspect pathways with higher P-values."
        )
    else:
        df_top = df_sig.head(top_n).copy()
        if "Adjusted P-value" in df_top.columns and "Term" in df_top.columns:
            adj_p = df_top["Adjusted P-value"].fillna(1.0).clip(lower=1e-300)
            neg_log_fdr = -np.log10(adj_p.values)
            terms = [
                (str(t)[:44] + "...") if len(str(t)) > 47 else str(t)
                for t in df_top["Term"].tolist()
            ]

            fig, ax = plt.subplots(figsize=(6.8, max(2.6, 0.42 * len(df_top) + 0.9)))
            y_pos = np.arange(len(terms))[::-1]
            bars = ax.barh(y_pos, neg_log_fdr, height=0.62, color=bar_color, alpha=0.88)

            fdr_thresh = -np.log10(fdr_cutoff)
            ax.axvline(fdr_thresh, color="#475569", linestyle="--", linewidth=1.0, label=f"FDR = {fdr_cutoff}")

            max_x = float(np.max(neg_log_fdr)) if len(neg_log_fdr) > 0 else 1.0
            for bar, val in zip(bars, neg_log_fdr):
                ax.text(
                    val + max(max_x * 0.02, 0.05),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{val:.2f}",
                    va="center",
                    ha="left",
                    fontsize=8.2,
                    color="#1e1b4b",
                    fontweight="bold",
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(terms, fontsize=8.5)
            ax.set_xlabel("-log10(Adjusted P-value)", fontsize=9.0, fontweight="bold", color="#1e1b4b")
            ax.set_title(title, fontsize=10.0, fontweight="bold", color="#1e1b4b", pad=10)
            ax.set_xlim(0, max(max_x * 1.22, fdr_thresh * 1.15))
            ax.legend(loc="lower right", frameon=False, fontsize=8.0)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        col_cfg = {
            "Gene_set": st.column_config.TextColumn("Library"),
            "Term": st.column_config.TextColumn("Pathway Term"),
            "Overlap": st.column_config.TextColumn("Overlap"),
            "P-value": st.column_config.NumberColumn("P-value", format="%.2e"),
            "Adjusted P-value": st.column_config.NumberColumn("Adj. P-value", format="%.2e"),
            "Odds Ratio": st.column_config.NumberColumn("Odds Ratio", format="%.2f"),
            "Combined Score": st.column_config.NumberColumn("Combined Score", format="%.1f"),
            "Genes": st.column_config.TextColumn("Genes"),
        }
        st.dataframe(df_sig.head(20), column_config=col_cfg, width="stretch", hide_index=True)

    st.download_button(
        label="💾 Download Full TSV (All P-values)",
        data=df_lib.to_csv(sep="\t", index=False),
        file_name=download_filename,
        mime="text/tab-separated-values",
        key=f"dl_{download_filename}_{title}",
    )


# =============================================================================
# SPLIT SYNCHRONIZATION & COHORT STATISTICS OVERLAY MODAL
# =============================================================================
def resolve_split_script_path():
    """Locate 1_tissues_selection_and_split.py across expected project locations."""
    candidates = [
        os.path.join(
            getattr(config_file_tree, "SPLIT_SCRIPT_DIR", os.path.join(config_file_tree.ROOT_DIR, "4_train_test_split")),
            "1_tissues_selection_and_split.py",
        ),
        os.path.join(config_file_tree.ROOT_DIR, "4_train_test_split", "1_tissues_selection_and_split.py"),
        os.path.join(config_file_tree.ROOT_DIR, "1_tissues_selection_and_split.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def resolve_preprocessed_clinical_path():
    """Locate preprocessed_clinical_data.tsv from config_file_tree directories."""
    candidate_dirs = [
        getattr(config_file_tree, "PREPROCESSED_DATA_DIR", ""),
        getattr(config_file_tree, "PREPROCESSED_SAMPLES_DIR", ""),
        os.path.join(config_file_tree.ROOT_DIR, "2_preprocessing", "preprocessed_data", "data_HERCULES_DECIDER"),
    ]
    for d in candidate_dirs:
        if d:
            p = os.path.join(d, "preprocessed_clinical_data.tsv")
            if os.path.exists(p):
                return p
    base = candidate_dirs[0] or candidate_dirs[1] or config_file_tree.ROOT_DIR
    return os.path.join(base, "preprocessed_clinical_data.tsv")


def detect_disk_subcohort():
    """Infer which subcohort ('both', 'nact', 'pds', or None) is currently written in SPLIT_DIR."""
    split_dir = config_file_tree.SPLIT_DIR
    clin_train_path = os.path.join(split_dir, "clinical_data_for_training.tsv")
    train_val_path = os.path.join(split_dir, "samples_train_val.csv")
    test_path = os.path.join(split_dir, "samples_test.csv")

    if not (os.path.exists(clin_train_path) and os.path.exists(train_val_path) and os.path.exists(test_path)):
        return None

    df_clin = load_tsv(clin_train_path)
    if df_clin is not None and "Treatment_strategy" in df_clin.columns:
        strategies = {
            str(v).strip().lower()
            for v in df_clin["Treatment_strategy"].dropna().unique()
            if str(v).strip()
        }
        if strategies == {"nact"}:
            return "nact"
        if strategies == {"pds"}:
            return "pds"
        if "nact" in strategies and "pds" in strategies:
            return "both"
    return None


def execute_split_script(cohort_key):
    """Run 1_tissues_selection_and_split.py for the given subcohort and return (success, logs)."""
    split_script_path = resolve_split_script_path()
    env = os.environ.copy()
    env["PYTHONPATH"] = config_file_tree.ROOT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    split_cmd = [sys.executable, "-u", split_script_path, "--subcohort", cohort_key]
    split_proc = subprocess.run(
        split_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=config_file_tree.ROOT_DIR,
        env=env,
    )
    return split_proc.returncode == 0, (split_proc.stdout or "")


def ensure_subcohort_split_synced(cohort_key):
    """Automatically regenerate split files whenever the active subcohort changes."""
    synced_key = st.session_state.get("synced_cohort_key")
    if synced_key != cohort_key:
        if synced_key is None and detect_disk_subcohort() == cohort_key:
            st.session_state["synced_cohort_key"] = cohort_key
        else:
            with st.spinner(f"Updating dataset split for {cohort_key.upper()}..."):
                ok, logs = execute_split_script(cohort_key)
            st.session_state["last_split_logs"] = logs
            if not ok:
                st.sidebar.error(f"❌ Split failed ({cohort_key.upper()})")
                with st.sidebar.expander("Split Error Logs"):
                    st.code(logs)
                return False
            st.session_state["synced_cohort_key"] = cohort_key

    st.sidebar.success(f"✅ Split synced ({cohort_key.upper()})")
    return True


@st.cache_data
def load_cohort_split_datasets(clin_train_path, clin_preproc_path, split_dir, mtimes_sig):
    """Load clinical_data_for_training.tsv for cohort/split stats and preprocessed_clinical_data.tsv for feature exploration."""
    df_train = load_tsv(clin_train_path)
    if df_train is None or df_train.empty:
        return None

    pid_col = "patient_id" if "patient_id" in df_train.columns else df_train.columns[0]
    df_train = df_train.copy()
    df_train[pid_col] = df_train[pid_col].astype(str).str.strip()
    df_train = df_train.drop_duplicates(subset=[pid_col]).reset_index(drop=True)
    selected_pids = set(df_train[pid_col])

    df_meta_train = load_tsv(os.path.join(split_dir, "metadata_for_training.tsv"))
    sample_to_patient = {}
    if df_meta_train is not None and {"sample_id", "patient_id"}.issubset(df_meta_train.columns):
        sample_to_patient = dict(
            zip(
                df_meta_train["sample_id"].astype(str).str.strip(),
                df_meta_train["patient_id"].astype(str).str.strip(),
            )
        )

    def _extract_split_pids(csv_path):
        df_sp = load_tsv(csv_path)
        if df_sp is None or df_sp.empty:
            return set()
        if "patient_id" in df_sp.columns:
            pids = df_sp["patient_id"].dropna().astype(str).str.strip()
        elif "sample_id" in df_sp.columns and sample_to_patient:
            pids = df_sp["sample_id"].astype(str).str.strip().map(sample_to_patient).dropna()
        else:
            return set()
        return set(pids) & selected_pids

    train_val_pids = _extract_split_pids(os.path.join(split_dir, "samples_train_val.csv"))
    test_pids = _extract_split_pids(os.path.join(split_dir, "samples_test.csv"))

    df_preproc_all = load_tsv(clin_preproc_path)
    preproc_pid_col = pid_col
    df_preproc_sel = None
    df_preproc_tv = None
    df_preproc_test = None
    if df_preproc_all is not None and not df_preproc_all.empty:
        preproc_pid_col = "patient_id" if "patient_id" in df_preproc_all.columns else df_preproc_all.columns[0]
        df_preproc_all = df_preproc_all.copy()
        df_preproc_all[preproc_pid_col] = df_preproc_all[preproc_pid_col].astype(str).str.strip()
        df_preproc_all = df_preproc_all.drop_duplicates(subset=[preproc_pid_col]).reset_index(drop=True)
        df_preproc_sel = df_preproc_all[df_preproc_all[preproc_pid_col].isin(selected_pids)].copy()
        df_preproc_tv = df_preproc_all[df_preproc_all[preproc_pid_col].isin(train_val_pids)].copy()
        df_preproc_test = df_preproc_all[df_preproc_all[preproc_pid_col].isin(test_pids)].copy()

    return {
        "pid_col": pid_col,
        "selected": df_train,
        "train_val": df_train[df_train[pid_col].isin(train_val_pids)].copy(),
        "test": df_train[df_train[pid_col].isin(test_pids)].copy(),
        "preproc_pid_col": preproc_pid_col,
        "preproc_all": df_preproc_all,
        "preproc_selected": df_preproc_sel,
        "preproc_train_val": df_preproc_tv,
        "preproc_test": df_preproc_test,
    }


def get_active_cohort_datasets():
    """Resolve paths and load cached training + preprocessed clinical tables for the active subcohort."""
    split_dir = config_file_tree.SPLIT_DIR
    clin_train_path = os.path.join(split_dir, "clinical_data_for_training.tsv")
    clin_preproc_path = resolve_preprocessed_clinical_path()

    tracked_files = [
        clin_train_path,
        clin_preproc_path,
        os.path.join(split_dir, "samples_train_val.csv"),
        os.path.join(split_dir, "samples_test.csv"),
    ]
    mtimes_sig = tuple(os.path.getmtime(p) if os.path.exists(p) else -1.0 for p in tracked_files)
    data = load_cohort_split_datasets(clin_train_path, clin_preproc_path, split_dir, mtimes_sig)
    return data, clin_train_path, clin_preproc_path


def _render_clinical_feature_explorer(subsets_dict, df_ref, pid_col, key_prefix, select_label, title_suffix):
    """Render a dropdown + segmented distribution plot/table for clinical features in df_ref."""
    feature_candidates = [c for c in df_ref.columns if c != pid_col]
    preferred = [
        c for c in [
            "Treatment_strategy", "Age_at_Diagnosis", "Stage_FIGO2014", "Histology",
            "Residual_tumor", "Primary_therapy_outcome", "is_platinum_resistant",
            "is_hrd", "BRCA_mutation_any", "OS_KaplanM_allHGSC", "PFS_KaplanM_allHGSC",
        ] if c in feature_candidates
    ]
    ordered_features = preferred + [c for c in feature_candidates if c not in preferred]

    selected_feature = st.selectbox(
        select_label,
        options=ordered_features,
        index=0,
        key=f"{key_prefix}_feature_select",
    )

    _render_categorical_comparison(
        subsets_dict,
        col_name=selected_feature,
        chart_title=f"Distribution of '{selected_feature}' ({title_suffix})",
    )


@st.dialog("📊 Patient Subcohort & Split Statistics", width="large")
def show_cohort_statistics_dialog(cohort_label, cohort_key, default_target_col):
    """Overlay modal displaying split counts/target and clinical feature explorers."""
    data, clin_train_path, clin_preproc_path = get_active_cohort_datasets()

    st.markdown(
        f"Active Subcohort: <code>{cohort_label}</code> | "
        f"Training Clinical Source: {path_badge(clin_train_path, mode='short')}",
        unsafe_allow_html=True,
    )

    if data is None:
        show_path_alert("error", "❌ Training clinical file not found or empty at:", clin_train_path)
        return

    df_selected = data["selected"]
    df_train_val = data["train_val"]
    df_test = data["test"]

    st.subheader("1. Total Patient Counts")
    n_sel = len(df_selected)
    tv_pct = (100.0 * len(df_train_val) / n_sel) if n_sel > 0 else 0.0
    test_pct = (100.0 * len(df_test) / n_sel) if n_sel > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Active Cohort ({cohort_key.upper()})", f"{n_sel} patients")
    c2.metric("Train + Val Split", f"{len(df_train_val)} ({tv_pct:.1f}%)")
    c3.metric("Test Split", f"{len(df_test)} ({test_pct:.1f}%)")

    train_subsets = {
        f"Active Cohort ({cohort_key.upper()})": df_selected,
        "Train + Val Split": df_train_val,
        "Test Split": df_test,
    }

    st.markdown("---")
    target_col = default_target_col if default_target_col in df_selected.columns else (
        "is_platinum_resistant" if "is_platinum_resistant" in df_selected.columns else df_selected.columns[-1]
    )
    st.subheader(f"2. Target Class Distribution (`{target_col}`)")
    _render_categorical_comparison(
        train_subsets,
        col_name=target_col,
        chart_title=f"Target Class Distribution ({target_col}) — Active Cohort",
    )

    st.markdown("---")
    st.subheader("3. Clinical Feature Distribution Explorer (Active Cohort)")
    if data["preproc_selected"] is not None:
        preproc_cohort_subsets = {
            f"Active Cohort ({cohort_key.upper()})": data["preproc_selected"],
            "Train + Val Split": data["preproc_train_val"],
            "Test Split": data["preproc_test"],
        }
        _render_clinical_feature_explorer(
            subsets_dict=preproc_cohort_subsets,
            df_ref=data["preproc_selected"],
            pid_col=data["preproc_pid_col"],
            key_prefix="modal_cohort_preproc",
            select_label="Select a feature from `preprocessed_clinical_data.tsv` (restricted to active training patients):",
            title_suffix="Active Cohort Splits",
        )
    else:
        show_path_alert("warning", "⚠️ Preprocessed clinical file not found at:", clin_preproc_path)

    st.markdown("---")
    st.subheader("4. Full Preprocessed Clinical Data Explorer (All Patients)")
    if data["preproc_all"] is not None:
        _render_clinical_feature_explorer(
            subsets_dict={"Full Preprocessed": data["preproc_all"]},
            df_ref=data["preproc_all"],
            pid_col=data["preproc_pid_col"],
            key_prefix="modal_full_preproc",
            select_label="Select a feature from `preprocessed_clinical_data.tsv` (full preprocessed dataset):",
            title_suffix="Full Preprocessed Dataset",
        )
    else:
        show_path_alert("warning", "⚠️ Preprocessed clinical file not found at:", clin_preproc_path)


def _clean_clinical_cell(val):
    """Normalize raw clinical TSV values (including stringified Python lists)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na"}:
        return np.nan
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                cleaned_items = [str(x).strip() for x in parsed if str(x).strip()]
                return ", ".join(cleaned_items) if cleaned_items else "None"
        except Exception:
            pass
    return s


def _prepare_categorical_or_binned_series(subsets_dict, col_name, n_bins=5):
    """Clean categorical values or discretize continuous numeric features (>6 unique values) into regular intervals."""
    raw_cleaned = {
        label: (df[col_name].map(_clean_clinical_cell) if col_name in df.columns else pd.Series([np.nan] * len(df)))
        for label, df in subsets_dict.items()
    }

    non_null = pd.concat([s.dropna() for s in raw_cleaned.values()], ignore_index=True) if raw_cleaned else pd.Series(dtype=object)
    num_all = pd.to_numeric(non_null, errors="coerce")
    is_numeric = not non_null.empty and num_all.notna().mean() >= 0.85
    valid_nums = num_all.dropna()

    if is_numeric and valid_nums.nunique() > 6:
        vmin, vmax = float(valid_nums.min()), float(valid_nums.max())
        edges = np.linspace(vmin, vmax, n_bins + 1)
        is_int = bool(np.all(np.isclose(valid_nums, np.round(valid_nums))))
        fmt = "{:.0f} – {:.0f}" if is_int else "{:.1f} – {:.1f}"
        bin_labels = [fmt.format(edges[i], edges[i + 1]) for i in range(n_bins)]

        binned_series = {}
        for label, s in raw_cleaned.items():
            s_num = pd.to_numeric(s, errors="coerce")
            cats = pd.cut(s_num, bins=edges, labels=bin_labels, include_lowest=True)
            binned_series[label] = cats.astype(object).fillna("Missing")

        has_missing = any((s == "Missing").any() for s in binned_series.values())
        ordered_cats = bin_labels + (["Missing"] if has_missing else [])
        return binned_series, ordered_cats

    cleaned_series = {}
    all_cats = set()
    for label, s in raw_cleaned.items():
        if is_numeric:
            def _fmt_discrete(v):
                if pd.isna(v):
                    return "Missing"
                fv = float(v)
                return str(int(round(fv))) if np.isclose(fv, round(fv)) else f"{fv:g}"
            s_clean = pd.to_numeric(s, errors="coerce").map(_fmt_discrete)
        else:
            s_clean = s.fillna("Missing").astype(str)
        cleaned_series[label] = s_clean
        all_cats.update(s_clean.unique().tolist())

    non_missing = [c for c in all_cats if c != "Missing"]
    ordered_cats = (
        sorted(non_missing, key=lambda x: float(x)) if is_numeric else sorted(non_missing)
    ) + (["Missing"] if "Missing" in all_cats else [])
    return cleaned_series, ordered_cats


def _render_categorical_comparison(subsets_dict, col_name, chart_title):
    """Render a horizontal segmented bar (0 to N) with legend and full split table below."""
    palette = [
        "#7c3aed", "#0d9488", "#2563eb", "#d97706", "#db2777",
        "#059669", "#4f46e5", "#e11d48", "#0284c7", "#ca8a04",
        "#9333ea", "#16a34a",
    ]

    cleaned_series, ordered_cats = _prepare_categorical_or_binned_series(subsets_dict, col_name)

    table_rows = []
    for cat in ordered_cats:
        row = {"Category": str(cat)}
        for label, s in cleaned_series.items():
            cnt = int((s == cat).sum())
            total = len(s)
            pct = (100.0 * cnt / total) if total > 0 else 0.0
            row[f"{label} (n={total})"] = f"{cnt} ({pct:.1f}%)"
        table_rows.append(row)
    df_summary = pd.DataFrame(table_rows)

    primary_label, primary_s = next(iter(cleaned_series.items()))
    total_n = len(primary_s)

    if total_n > 0:
        active_items = [(idx, cat, int((primary_s == cat).sum())) for idx, cat in enumerate(ordered_cats)]
        active_items = [(idx, cat, cnt) for idx, cat, cnt in active_items if cnt > 0]

        n_items = max(len(active_items), 1)
        avg_lbl_len = np.mean([len(f"{cat}: {cnt} (100.0%)") for _, cat, cnt in active_items]) if active_items else 12
        ncol = min(n_items, 3 if avg_lbl_len > 19 else 4)
        n_rows = (n_items + ncol - 1) // ncol

        fig_h = 1.65 + 0.26 * n_rows
        title_pad = 14 + n_rows * 17
        edge_lw = 0.8 if n_items > 6 else 1.8

        fig, ax = plt.subplots(figsize=(8.4, fig_h))
        left = 0
        split_ticks = [0]

        for idx, cat, cnt in active_items:
            pct = 100.0 * cnt / total_n
            color = "#94a3b8" if cat == "Missing" else palette[idx % len(palette)]

            ax.barh(
                [0],
                [cnt],
                left=left,
                height=0.52,
                color=color,
                edgecolor="white",
                linewidth=edge_lw,
                label=f"{cat}: {cnt} ({pct:.1f}%)",
            )

            if pct >= 12.0:
                lbl_txt = f"{cat}\n {pct:.1f}%" if len(str(cat)) <= 16 else f" {pct:.1f}%"
                ax.text(
                    left + cnt / 2.0,
                    0,
                    lbl_txt,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.6,
                    fontweight="bold",
                )

            left += cnt
            if left < total_n and (left - split_ticks[-1]) >= max(1, total_n * 0.045):
                split_ticks.append(left)

        if split_ticks[-1] != total_n:
            if len(split_ticks) > 1 and (total_n - split_ticks[-1]) < total_n * 0.045:
                split_ticks[-1] = total_n
            else:
                split_ticks.append(total_n)

        ax.set_xlim(0, total_n)
        ax.set_ylim(-0.48, 0.48)
        ax.set_yticks([])
        ax.set_xticks(split_ticks)
        ax.tick_params(axis="x", labelsize=9.2, length=5, width=1.1, colors="#1e1b4b")
        for tick_lbl in ax.get_xticklabels():
            tick_lbl.set_fontweight("bold")
        ax.set_xlabel(f"Patients ({primary_label})", fontsize=9.0, fontweight="bold", color="#1e1b4b")

        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#475569")
        ax.spines["bottom"].set_linewidth(1.1)

        ax.set_title(chart_title, fontsize=10.2, fontweight="bold", color="#1e1b4b", pad=title_pad)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=ncol,
            frameon=False,
            fontsize=8.2,
            handlelength=1.2,
            columnspacing=1.1,
            labelspacing=0.35,
        )

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.dataframe(df_summary, width="stretch", hide_index=True)


def render_gene_pathway_box(tab_prefix):
    """Render the collapsible 'Change Displayed Genes' box inside a tab."""
    combined_groups = get_available_gene_groups()
    if not combined_groups:
        combined_groups = {"All Genes (Empty)": []}

    select_key = f"select_group_{tab_prefix}"
    custom_key = f"custom_gene_search_{tab_prefix}"
    edit_toggle_key = f"show_edit_pathway_{tab_prefix}"
    create_toggle_key = f"show_create_pathway_{tab_prefix}"

    if select_key in st.session_state and st.session_state[select_key] not in combined_groups:
        del st.session_state[select_key]

    def _on_pathway_change():
        st.session_state[custom_key] = ""

    with st.expander("Change Displayed Genes", expanded=False):
        col_group, col_edit_btn = st.columns([3, 1.2])
        with col_group:
            selected_group = st.selectbox(
                "Select Biological Group / Pathway:",
                options=list(combined_groups.keys()),
                key=select_key,
                on_change=_on_pathway_change,
            )
        with col_edit_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("✏️ Edit Selected Pathway", key=f"btn_toggle_edit_{tab_prefix}", width="stretch"):
                st.session_state[edit_toggle_key] = not st.session_state.get(edit_toggle_key, False)

        preset_genes = combined_groups.get(selected_group, [])

        if st.session_state.get(edit_toggle_key, False):
            with st.container(border=True):
                edit_genes_text = st.text_area(
                    f"Edit Genes in '{selected_group}' (remove or add comma-separated symbols):",
                    value=", ".join(preset_genes),
                    key=f"edit_genes_text_{tab_prefix}_{selected_group}",
                )
                col_save, col_del, _ = st.columns([1.2, 1.2, 2])
                if col_save.button("💾 Save Pathway Changes", key=f"btn_save_pathway_{tab_prefix}", width="stretch"):
                    raw_data = load_user_subsets()
                    deleted_defaults = [d for d in raw_data.get(DELETED_DEFAULTS_KEY, []) if d != selected_group]
                    if deleted_defaults:
                        raw_data[DELETED_DEFAULTS_KEY] = deleted_defaults
                    else:
                        raw_data.pop(DELETED_DEFAULTS_KEY, None)
                    raw_data[selected_group] = [g.strip().upper() for g in edit_genes_text.split(",") if g.strip()]
                    if save_user_subsets(raw_data):
                        st.session_state[edit_toggle_key] = False
                        st.success(f"Updated pathway '{selected_group}'!")
                        st.rerun()

                if col_del.button("🗑️ Delete Pathway", key=f"btn_delete_pathway_{tab_prefix}", width="stretch"):
                    raw_data = load_user_subsets()
                    raw_data.pop(selected_group, None)
                    if selected_group in DEFAULT_GENE_GROUPS:
                        deleted_defaults = set(raw_data.get(DELETED_DEFAULTS_KEY, []))
                        deleted_defaults.add(selected_group)
                        raw_data[DELETED_DEFAULTS_KEY] = sorted(deleted_defaults)
                    if save_user_subsets(raw_data):
                        st.session_state[edit_toggle_key] = False
                        st.session_state.pop(select_key, None)
                        st.success(f"Deleted pathway '{selected_group}'!")
                        st.rerun()

        if preset_genes:
            st.success(f"**Included Genes ({len(preset_genes)}):** {', '.join(preset_genes)}")

        if st.button("➕ Create New Pathway", key=f"btn_toggle_create_{tab_prefix}"):
            st.session_state[create_toggle_key] = not st.session_state.get(create_toggle_key, False)

        if st.session_state.get(create_toggle_key, False):
            with st.container(border=True):
                new_subset_name = st.text_input(
                    "New Pathway Name:",
                    placeholder="e.g. Immune Response Genes",
                    key=f"new_subset_name_{tab_prefix}",
                )
                new_subset_genes = st.text_input(
                    "New Pathway Genes (comma separated):",
                    placeholder="e.g. TP53, BRCA1, EGFR",
                    key=f"new_subset_genes_{tab_prefix}",
                )
                if st.button("💾 Save New Pathway", key=f"btn_create_pathway_{tab_prefix}"):
                    clean_name = new_subset_name.strip()
                    if clean_name and new_subset_genes.strip():
                        raw_data = load_user_subsets()
                        deleted_defaults = [d for d in raw_data.get(DELETED_DEFAULTS_KEY, []) if d != clean_name]
                        if deleted_defaults:
                            raw_data[DELETED_DEFAULTS_KEY] = deleted_defaults
                        else:
                            raw_data.pop(DELETED_DEFAULTS_KEY, None)
                        raw_data[clean_name] = [g.strip().upper() for g in new_subset_genes.split(",") if g.strip()]
                        if save_user_subsets(raw_data):
                            st.session_state[create_toggle_key] = False
                            st.session_state[select_key] = clean_name
                            st.session_state[custom_key] = ""
                            st.success(f"Created pathway '{clean_name}'!")
                            st.rerun()

        custom_gene_search = st.text_input(
            "Or type specific Gene Symbols (comma separated):",
            placeholder="e.g. TP53, APOC3, BRCA1",
            key=custom_key,
        )

        return (
            [g.strip().upper() for g in custom_gene_search.split(",") if g.strip()]
            if custom_gene_search.strip()
            else preset_genes
        )


# =============================================================================
# INTERACTIVE PPI NETWORK BUILDER
# =============================================================================
@st.cache_data
def build_optimized_pyvis_subgraph(
    df_edges,
    target_genes_tuple=None,
    top_n=100,
    max_neighbors=5,
    saliency_df=None,
    color_mode="Saliency Score",
    enable_physics=False,
    out_dir=None,
):
    """Build an interactive PyVis PPI network centered on top edges or selected target genes."""
    if df_edges is None or df_edges.empty:
        return None

    target_set = {str(g).upper() for g in target_genes_tuple} if target_genes_tuple else set()

    if target_set:
        gene_a = df_edges["gene_symbol_a"].astype(str).str.upper()
        gene_b = df_edges["gene_symbol_b"].astype(str).str.upper()
        selected_indices = set()
        for gene in target_set:
            top_gene_edges = df_edges[(gene_a == gene) | (gene_b == gene)].sort_values(
                by="attention_weight", ascending=False
            ).head(max_neighbors)
            selected_indices.update(top_gene_edges.index.tolist())
        df_sub = (
            df_edges.loc[list(selected_indices)].sort_values(by="attention_weight", ascending=False)
            if selected_indices
            else pd.DataFrame(columns=df_edges.columns)
        )
    else:
        df_sub = df_edges.head(top_n)

    G = nx.Graph()
    saliency_dict = (
        dict(zip(saliency_df["gene_name"], saliency_df["saliency_score"]))
        if saliency_df is not None and not saliency_df.empty
        else {}
    )

    min_w = df_sub["attention_weight"].min() if not df_sub.empty else 0.0
    max_w = df_sub["attention_weight"].max() if not df_sub.empty else 1.0
    denom_w = (max_w - min_w) or 1.0

    for _, row in df_sub.iterrows():
        u, v = str(row["gene_symbol_a"]), str(row["gene_symbol_b"])
        raw_w = float(row["attention_weight"])
        G.add_edge(
            u,
            v,
            weight=raw_w,
            width=2.0 + 4.0 * ((raw_w - min_w) / denom_w),
            title=f"Interaction: {u} - {v}\nModel Interaction Weight: {raw_w:.5f}",
            color={"color": "rgba(80, 120, 170, 0.75)", "highlight": "#e63946"},
        )

    for tg in target_set:
        if any(str(node).upper() == tg for node in G.nodes()):
            continue
        display_name = tg
        if saliency_df is not None and "gene_name" in saliency_df.columns:
            matches = saliency_df[saliency_df["gene_name"].astype(str).str.upper() == tg]
            if not matches.empty:
                display_name = str(matches.iloc[0]["gene_name"])
        G.add_node(display_name)

    for node, (x, y) in nx.spring_layout(G, k=0.18, seed=42).items():
        G.nodes[node]["x"] = float(x * 1000)
        G.nodes[node]["y"] = float(y * 1000)

    degrees = dict(G.degree())
    min_deg, max_deg = (min(degrees.values()), max(degrees.values())) if degrees else (0, 1)
    comp_map = {node: idx for idx, comp in enumerate(nx.connected_components(G)) for node in comp}

    sal_logs = [np.log10(max(saliency_dict.get(n, 1e-6), 1e-8)) for n in G.nodes()]
    min_log_sal = min(sal_logs) if sal_logs else -8.0
    denom_log_sal = ((max(sal_logs) if sal_logs else 0.0) - min_log_sal) or 1.0

    for node in G.nodes():
        deg = degrees[node]
        comp_idx = comp_map.get(node, 0)
        sal_score = saliency_dict.get(node, 1e-6)
        norm_deg = (deg - min_deg) / (max_deg - min_deg) if max_deg != min_deg else 0.5
        is_target = str(node).upper() in target_set

        if color_mode == "Node Degree (Number of Connections)":
            hex_color = mcolors.to_hex(plt.colormaps["plasma"](0.15 + 0.8 * norm_deg))
            tooltip = f"Gene: {node}\nConnections: {deg}\nGene Importance Score: {sal_score:.6f}"
            node_size = 14 + 18 * norm_deg
        elif color_mode == "Connected Component (Clusters)":
            hex_color = mcolors.to_hex(plt.colormaps["Set3"](comp_idx % 12))
            tooltip = f"Gene: {node}\nNetwork Module: #{comp_idx + 1}\nConnections: {deg}"
            node_size = 16 + 14 * norm_deg
        else:
            norm_sal = (np.log10(max(sal_score, 1e-8)) - min_log_sal) / denom_log_sal
            hex_color = mcolors.to_hex(plt.colormaps["Spectral_r"](norm_sal))
            tooltip = (
                f"Gene: {node}\nGene Importance Score: {sal_score:.6f}\n"
                f"Relative Percentile: {norm_sal * 100:.1f}%\nConnections: {deg}"
            )
            node_size = 14 + 18 * float(norm_sal)

        if is_target:
            tooltip = f"🌟 [SELECTED TARGET GENE]\n{tooltip}"
            node_size = max(node_size * 1.25, 22.0)
            G.nodes[node]["borderWidth"] = 5
            G.nodes[node]["shadow"] = {"enabled": True, "color": "rgba(230, 57, 70, 0.85)", "size": 20, "x": 0, "y": 0}
            G.nodes[node]["color"] = {
                "background": hex_color,
                "border": "#E63946",
                "highlight": {"background": "#FFD166", "border": "#D90429"},
            }
        else:
            G.nodes[node]["borderWidth"] = 1
            G.nodes[node]["shadow"] = {"enabled": False}
            G.nodes[node]["color"] = {
                "background": hex_color,
                "border": "#333333",
                "highlight": {"background": "#A8DDA8", "border": "#111111"},
            }

        G.nodes[node]["size"] = node_size
        G.nodes[node]["font"] = {
            "color": "#111111",
            "size": 14 if is_target else 12,
            "face": "arial",
            "bold": is_target,
        }
        G.nodes[node]["title"] = tooltip

    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=False)
    net.from_nx(G)
    if not enable_physics:
        net.toggle_physics(False)

    os.makedirs(out_dir or ".", exist_ok=True)
    html_path = os.path.join(out_dir or ".", "temp_network.html")
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    if "</body>" in html_content:
        return html_content.replace("</body>", f"{_NETWORK_SEARCH_OVERLAY_HTML}\n</body>")
    return html_content + _NETWORK_SEARCH_OVERLAY_HTML


# =============================================================================
# PIPELINE SUBPROCESS RUNNER
# =============================================================================
def run_explainability_step(selected_model_name, step_name="all", cohort_key="both", is_lightweight=False):
    """Execute the requested explainability step (and any required upstream steps) via subprocess."""
    is_gat = "GAT" in selected_model_name
    if is_gat:
        target_script = "explainability_gat_samples.py"
    elif is_lightweight:
        target_script = "explainability_linear_mil.py"
    else:
        target_script = "explainability_mil_patients.py"

    script_path = os.path.join(config_file_tree.EXPLAINABILITY_DIR, target_script)
    if not os.path.exists(script_path):
        script_path = os.path.join(config_file_tree.ROOT_DIR, target_script)

    env = os.environ.copy()
    env["PYTHONPATH"] = config_file_tree.ROOT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    banner_box = st.empty()

    banner_box.info(f"Running explainability analysis (`{step_name}`) via `{to_rel_path(script_path)}`...")
    status_text.text(f"⏳ Running analysis step: {step_name}...")

    cmd = [sys.executable, "-u", script_path, "--step", step_name]
    if is_lightweight and not is_gat:
        mil_lw_dir = st.session_state.get("custom_linear_mil_path", "")
        cmd.extend([
            "--run_folder", mil_lw_dir,
            "--cohort", st.session_state.get("lw_eval_cohort", "oof"),
            "--classifier", st.session_state.get("lw_classifier", "elasticnet"),
        ])
        if st.session_state.get("custom_gat_path"):
            cmd.extend(["--gat_model_path", os.path.dirname(st.session_state["custom_gat_path"])])
    else:
        if st.session_state.get("custom_gat_path"):
            cmd.extend(["--gat_model_path", st.session_state["custom_gat_path"]])
        if not is_gat and st.session_state.get("custom_mil_path"):
            cmd.extend(["--mil_model_path", st.session_state["custom_mil_path"]])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=config_file_tree.ROOT_DIR,
        env=env,
        bufsize=1,
    )

    stdout_logs = [f"=== Explainability ({step_name} | {cohort_key.upper()}) ===\n"]
    for line in iter(process.stdout.readline, ""):
        if line.startswith("[PROGRESS]"):
            try:
                pct_str, msg = line[len("[PROGRESS]"):].strip().split("|", 1)
                pct_val = max(0.0, min(1.0, float(pct_str.strip())))
                progress_bar.progress(pct_val)
                status_text.text(f"⏳ [{int(pct_val * 100)}%] {msg.strip()}")
            except Exception:
                pass
        else:
            stdout_logs.append(line)

    process.stdout.close()
    if process.wait() == 0:
        progress_bar.progress(1.0)
        status_text.empty()
        banner_box.success(f"Analysis `{step_name}` ({cohort_key.upper()}) completed successfully!")
        with st.expander("Show Execution Logs"):
            st.code("".join(stdout_logs))
    else:
        progress_bar.empty()
        status_text.empty()
        banner_box.error(f"Error running step `{step_name}` in `{target_script}`")
        with st.expander("Show Execution Logs"):
            st.code("".join(stdout_logs))


# =============================================================================
# SIDEBAR: PATIENT SUBCOHORT & MODEL SELECTION
# =============================================================================
st.sidebar.title("🧬 HGSC Explainable AI Navigator")
st.sidebar.caption("Shows what trained models learned and prioritized to make predictions — not causal biological proof.")
st.sidebar.markdown("---")

st.sidebar.subheader(
    "👨‍👨‍👦Patient Subcohort",
    help="Select the patient treatment cohort ('Both', 'NACT', or 'PDS'). Automatically filters available trained models and regenerates dataset subdivision.",
)
selected_cohort_label = st.sidebar.radio(
    "Active Patient Subcohort:",
    options=list(COHORT_OPTIONS.keys()),
    index=0,
    horizontal=True,
)
cohort_summary_box = st.sidebar.empty()
selected_cohort_key = COHORT_OPTIONS[selected_cohort_label]

ensure_subcohort_split_synced(selected_cohort_key)

open_cohort_stats_clicked = st.sidebar.button(
    "📊 Inspect Cohort & Dataset Statistics",
    width="stretch",
    help="Open an overlay window showing patient counts, target class distributions, and clinical feature distributions across Global Preprocessed, Training/Validation dataset, and Test dataset.",
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📂 Load Models to Display",
    help=(
        "Select an experiment directory matching the active subcohort, then choose which "
        "saved model to inspect."
    ),
)

experiments = find_experiment_dirs(selected_cohort_key)
exp_map = {item[0]: item for item in experiments}
selected_exp_name, mil_pt, linear_mil_dir, gat_pt = None, None, None, None

if exp_map:
    exp_names = list(exp_map.keys())
    default_folder = getattr(config_file_tree, "PATIENT_ENCODER_FOLDER", "")
    default_idx = next(
        (i for i, name in enumerate(exp_names) if default_folder and name in default_folder), 0
    )
    selected_exp_name = st.sidebar.selectbox(
        "Experiment Directory:",
        options=exp_names,
        index=default_idx,
        key="selected_exp_name_box",
    )
    _, mil_pt, linear_mil_dir, gat_pt = exp_map[selected_exp_name]
else:
    st.sidebar.warning(f"No trained model directories matching '{selected_cohort_label}' found.")

if mil_pt:
    st.session_state["custom_mil_path"] = mil_pt
else:
    st.session_state.pop("custom_mil_path", None)

if linear_mil_dir:
    st.session_state["custom_linear_mil_path"] = linear_mil_dir
else:
    st.session_state.pop("custom_linear_mil_path", None)

resolved_gat_pt, is_redirected_gat = resolve_original_gat_checkpoint(gat_pt)
if resolved_gat_pt:
    st.session_state["custom_gat_path"] = resolved_gat_pt
    st.session_state["is_redirected_gat"] = is_redirected_gat
else:
    st.session_state.pop("custom_gat_path", None)
    st.session_state["is_redirected_gat"] = False

# Build ordered model level options (Patient Level always first)
available_radio_options = []
if mil_pt:
    available_radio_options.append("Patient Level (MIL)")
if linear_mil_dir:
    available_radio_options.append("Patient Level (LinearMIL)")
if resolved_gat_pt:
    available_radio_options.append("Sample Level (GAT)")

if not available_radio_options:
    available_radio_options = ["Patient Level (MIL)", "Patient Level (LinearMIL)", "Sample Level (GAT)"]

# Always select the first radio option (Patient Level) whenever the experiment directory changes
if (
    st.session_state.get("_last_selected_exp") != selected_exp_name
    or st.session_state.get("selected_model_radio") not in available_radio_options
):
    st.session_state["_last_selected_exp"] = selected_exp_name
    st.session_state["selected_model_radio"] = available_radio_options[0]

selected_model = st.sidebar.radio(
    "Model Level to Display:",
    options=available_radio_options,
    key="selected_model_radio",
)

cfg = get_model_config(selected_model, selected_exp_name=selected_exp_name)
cohort_data, _, _ = get_active_cohort_datasets()
if cohort_data is not None:
    df_sel = cohort_data["selected"]
    t_col = (
        cfg["target_col"]
        if cfg["target_col"] in df_sel.columns
        else (
            config_file_tree.PATIENT_TARGET_COL
            if config_file_tree.PATIENT_TARGET_COL in df_sel.columns
            else ("is_platinum_resistant" if "is_platinum_resistant" in df_sel.columns else None)
        )
    )
    lines = [f"<strong>[ {len(df_sel)} patients ]</strong>"]
    if t_col in df_sel.columns:
        notes = {"no": " (PFI > 180)", "yes": " (PFI < 180)"} if t_col == "is_platinum_resistant" else {}
        counts = df_sel[t_col].map(_clean_clinical_cell).fillna("Missing").value_counts()
        ordered_vals = sorted([k for k in counts.index if k != "Missing"]) + (["Missing"] if "Missing" in counts.index else [])
        lines.extend(f"{int(counts[v])} '{v}'{notes.get(str(v).lower(), '')}" for v in ordered_vals)

    cohort_summary_box.markdown(
        f"<div style='background-color: #ede9fe; border: 1.5px solid #c4b5fd; color: #1e1b4b; "
        f"padding: 8px 12px; border-radius: 8px; line-height: 1.55; margin: 4px 0 10px 0;'>"
        f"{'<br>'.join(lines)}</div>",
        unsafe_allow_html=True,
    )

if cfg["is_lightweight"]:
    col_lw1, col_lw2 = st.sidebar.columns(2)
    with col_lw1:
        st.selectbox(
            "Classifier:",
            options=["elasticnet", "plsda"],
            format_func=lambda x: "ElasticNet" if x == "elasticnet" else "PLS-DA",
            key="lw_classifier",
        )
    with col_lw2:
        st.selectbox(
            "Eval Subset:",
            options=["oof", "test"],
            format_func=lambda x: x.upper(),
            key="lw_eval_cohort",
        )

if open_cohort_stats_clicked:
    show_cohort_statistics_dialog(selected_cohort_label, selected_cohort_key, cfg["target_col"])

run_pipeline_clicked = st.sidebar.button(
    "▶️ Run Explainability Pipeline",
    type="primary",
    width="stretch",
    disabled=not cfg["model_exists"],
    help="Generate all explanation plots and tables for the selected model and active subcohort.",
)

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Active File Trace")
if cfg["model_exists"]:
    ckpt_label = "fold_models (5-fold artifacts)" if cfg["is_lightweight"] else os.path.basename(cfg["model_path"])
    redir_note = (
        ' <span style="font-size: 0.85em; color: #6d28d9; font-weight: 600;">(Original Pretrained)</span>'
        if (cfg["prefix"] == "gat" and st.session_state.get("is_redirected_gat"))
        else ""
    )
    checkpoint_trace_html = (
        f'<span title="{to_rel_path(cfg["model_path"])}" style="cursor: help; '
        f'background-color: rgba(124, 58, 237, 0.12); color: #3b0764; padding: 2px 6px; border-radius: 4px;">'
        f'{ckpt_label}</span>{redir_note}'
    )
    results_trace_html = path_badge(
        cfg["explain_dir"],
        mode="short" if (cfg["prefix"] == "gat" and st.session_state.get("is_redirected_gat")) else "short",
    )
else:
    checkpoint_trace_html = '<span style="color: #991b1b; font-weight: 600;">⚠️ No saved model found</span>'
    results_trace_html = '<span style="color: #6b7280;">N/A</span>'

trace_rows = [
    f"<strong>Patient Subcohort:</strong> <i>{selected_cohort_label}</i>",
    f"<strong>Target Outcome:</strong> <i>{cfg['target_col']}</i>",
]
if cfg["is_lightweight"]:
    clf_disp = "PLS-DA" if st.session_state.get("lw_classifier", "elasticnet") == "plsda" else "ElasticNet"
    sub_disp = st.session_state.get("lw_eval_cohort", "oof").upper()
    trace_rows.append(f"<strong>Classifier / Subset:</strong> <i>{clf_disp}</i> / <i>{sub_disp}</i>")
trace_rows.extend([
    f"<strong>Model Checkpoint:</strong> <i>{checkpoint_trace_html}</i>",
    f"<strong>Results Folder:</strong> <i>{results_trace_html}</i>",
])

st.sidebar.markdown(
    f'<div style="background-color: rgba(124, 58, 237, 0.08); padding: 12px; border-radius: 8px; '
    f'border-left: 4px solid #7c3aed; line-height: 1.75; word-break: break-word;">'
    f'{"<br>".join(trace_rows)}</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# MAIN DASHBOARD AREA
# =============================================================================
st.title(f"📊 Model Explainability Dashboard — {selected_model}")
st.space("small")

if run_pipeline_clicked:
    run_explainability_step(
        selected_model, step_name="all", cohort_key=selected_cohort_key, is_lightweight=cfg["is_lightweight"]
    )

active_node_features = [
    cfg["node_feature_names"][i]
    for i in cfg["node_feature_indices"]
    if i < len(cfg["node_feature_names"])
]

tab_titles = [
    "📈 Evaluation & Performance Metrics",
    "🎯 Feature Importance & SHAP",
    "🕸️ PPI Graph & Gene Saliency Profiles",
    "🧬 Functional Pathway Enrichment",
]
if cfg["prefix"] == "mil":
    tab_titles.append("🧪 MIL Tissue Attention Distribution")
tab_titles.append("📉 Embedding Latent Space (PCA)")

tabs = st.tabs(tab_titles)
if cfg["prefix"] == "mil":
    tab_eval, tab_shap, tab_graph, tab_enrich, tab_tissue_attn, tab_pca = tabs
else:
    tab_eval, tab_shap, tab_graph, tab_enrich, tab_pca = tabs


# =============================================================================
# TAB 1: EVALUATION & PERFORMANCE METRICS
# =============================================================================
with tab_eval:
    st.header("Model Performance & Cross-Validation Results")
    st.markdown(
        "Summarizes how accurately the model predicts the target outcome across 5-fold cross-validation "
        "(splitting the cohort into 5 independent test subsets). Summary cards report the mean and variance across folds."
    )
    if st.button("▶️ Generate / Refresh Results for This Tab", key="btn_run_eval"):
        run_explainability_step(selected_model, step_name="importance", cohort_key=selected_cohort_key, is_lightweight=cfg["is_lightweight"])

    report_df, report_file_path = load_performance_report(
        cfg["model_dir"],
        cfg["prefix"],
        is_lightweight=cfg["is_lightweight"],
        classifier=st.session_state.get("lw_classifier", "elasticnet"),
    )
    df_clean_folds, summary_metrics = parse_report_metrics(report_df)

    if report_df is not None:
        rel_report_path = to_rel_path(report_file_path).replace('"', "&quot;")
        st.markdown(
            f'<div style="background-color: rgba(33, 195, 84, 0.1); color: #177233; '
            f'padding: 14px 16px; border-radius: 8px; margin-bottom: 16px; '
            f'font-size: 1.12rem; display: flex; align-items: center; gap: 8px;">'
            f'<span>✅ Loaded performance metrics report</span>'
            f'<span title="{rel_report_path}" style="display: inline-flex; align-items: center; '
            f'justify-content: center; width: 18px; height: 18px; border-radius: 50%; '
            f'border: 1.5px solid #177233; color: #177233; font-size: 0.78rem; '
            f'font-weight: 700; cursor: help; line-height: 1; opacity: 0.85;">?</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        if "auc" in summary_metrics:
            col1.metric("Mean AUC-ROC ± Var", summary_metrics["auc"]["display"])
        elif "c_index" in summary_metrics:
            col1.metric("Mean C-Index ± Var", summary_metrics["c_index"]["display"])
        if "f1" in summary_metrics:
            col2.metric("Mean Macro F1 ± Var", summary_metrics["f1"]["display"])
        if "acc" in summary_metrics:
            col3.metric("Mean Accuracy ± Var", summary_metrics["acc"]["display"])
        if "test_loss" in summary_metrics:
            col4.metric("Mean Test Loss ± Var", summary_metrics["test_loss"]["display"])

        st.markdown("---")
        st.subheader("📋 Per-Fold Validation Breakdown")
        st.caption("Inspect individual validation folds to verify that predictive performance is consistent across patient splits.")
        st.dataframe(df_clean_folds, width="stretch")
    else:
        show_path_alert("error", "❌ Performance report file not found at:", report_file_path)

    st.markdown("---")
    st.subheader("🎯 Sensitivity, Specificity & ROC Analysis")
    st.caption(
        "Evaluates Sensitivity (TPR), Specificity (TNR), Confusion Matrix, and ROC curves "
        "for the currently selected model."
    )

    metrics_dir = os.path.join(cfg["explain_dir"], "classification_metrics")
    sens_spec_png = get_explain_file(
        cfg["explain_dir"], "classification_metrics", f"{cfg['prefix']}_sensitivity_specificity_dashboard.png"
    )
    sens_spec_tsv = get_explain_file(
        cfg["explain_dir"], "classification_metrics", f"{cfg['prefix']}_sensitivity_specificity.tsv"
    )
    df_sens_spec = load_tsv(sens_spec_tsv)

    if os.path.exists(sens_spec_png) or df_sens_spec is not None:
        if os.path.exists(sens_spec_png):
            st.image(sens_spec_png, width="stretch")
        if df_sens_spec is not None:
            st.dataframe(df_sens_spec, width="stretch", hide_index=True)
    else:
        show_path_alert(
            "warning",
            "Sensitivity & Specificity results not found in:",
            metrics_dir,
            "- Click 'Generate / Refresh Results for This Tab' above to generate them.",
        )

    st.markdown("---")
    st.subheader(f"Predictive Relevance of {cfg['extra_type']}")
    st.caption(
        "Measures how much model accuracy drops when each clinical or sample variable is randomly shuffled—"
        "larger drops indicate stronger reliance on that variable."
    )
    meta_filepath = get_explain_file(cfg["explain_dir"], "feature_importance", get_importance_tsv_filename(cfg["prefix"]))
    df_meta_imp = load_tsv(meta_filepath)
    if df_meta_imp is not None:
        st.dataframe(df_meta_imp, width="stretch")
    else:
        show_path_alert("warning", "Feature importance table not found at:", meta_filepath, "- Click the button above to generate it.")


# =============================================================================
# TAB 2: SHAP & FEATURE IMPORTANCE
# =============================================================================
with tab_shap:
    st.header("Gene & Feature Impact on Predictions (SHAP Analysis)")
    st.markdown(
        "SHAP estimates how strongly each gene measurement or clinical variable pushes the model toward or away from "
        "the predicted outcome across individual samples or patients."
    )
    if st.button("▶️ Generate / Refresh Results for This Tab", key="btn_run_shap"):
        run_explainability_step(selected_model, step_name="shap", cohort_key=selected_cohort_key, is_lightweight=cfg["is_lightweight"])

    shap_genes = render_gene_pathway_box("shap")

    st.subheader("🎯 Molecular Modality SHAP Summaries")
    st.markdown("""
        **How to read these plots:**
        - **Horizontal Axis (SHAP Value):** Direction and magnitude of impact on the predicted outcome (positive values increase the predicted score; negative values decrease it).
        - **Dot Color (Measured Feature Value):** High values (e.g., high expression or copy number gain) appear in red/pink; low values appear in blue.
        - **Vertical Order:** Genes are ranked from top to bottom by overall impact across the cohort.
    """)
    st.space("small")

    shap_cols = st.columns(len(active_node_features)) if active_node_features else [st.container()]
    for idx, feat_name in enumerate(active_node_features):
        feat_shap_img = get_explain_file(
            cfg["explain_dir"], "shap_analysis", get_gene_shap_png_filename(cfg["prefix"], feat_name)
        )
        feat_shap_npz = get_explain_file(
            cfg["explain_dir"], "shap_analysis", get_gene_shap_npz_filename(cfg["prefix"], feat_name)
        )
        with shap_cols[idx % len(shap_cols)]:
            st.markdown(f"#### Modality: `{feat_name}`")
            st.space("small")
            if shap_genes:
                if os.path.exists(feat_shap_npz):
                    clean_feat = str(feat_name).replace(" ", "_").lower()
                    filtered_img_path = os.path.join(
                        cfg["explain_dir"],
                        "shap_analysis",
                        f"{cfg['prefix']}_gene_{clean_feat}_shap_summary_filtered.png",
                    )
                    os.makedirs(os.path.dirname(filtered_img_path), exist_ok=True)
                    ok, matched_cnt = generate_filtered_gene_shap_beeswarm(
                        npz_path=feat_shap_npz,
                        target_genes=shap_genes,
                        title=f"{cfg['prefix'].upper()} Selected Genes SHAP Summary ({feat_name})",
                        output_path=filtered_img_path,
                        max_display=len(shap_genes),
                    )
                    if ok:
                        st.image(
                            filtered_img_path,
                            width="stretch",
                            caption=f"Selected Genes ({matched_cnt}) SHAP Beeswarm for {feat_name}",
                        )
                    else:
                        st.info(f"No matching genes found for `{feat_name}` in the selected gene list.")
                else:
                    show_path_alert(
                        "warning",
                        f'Per-sample SHAP matrices for "{feat_name}" not found at:',
                        feat_shap_npz,
                        "- Click 'Generate / Refresh Results for This Tab' above to generate them.",
                    )
            elif os.path.exists(feat_shap_img):
                st.image(feat_shap_img, width="stretch", caption=f"Top Genes Impact Summary for {feat_name}")
            else:
                show_path_alert("warning", f'SHAP summary plot for "{feat_name}" not found at:', feat_shap_img)

    st.markdown("---")
    st.subheader("Combined Molecular & Clinical Features Summary")
    st.space("small")

    all_shap_img = get_explain_file(cfg["explain_dir"], "shap_analysis", get_all_shap_png_filename(cfg["prefix"]))
    if os.path.exists(all_shap_img):
        st.image(all_shap_img, width="stretch")
    else:
        show_path_alert("warning", "Combined SHAP summary plot not found at:", all_shap_img, "- Click the button above to run SHAP analysis.")

    if cfg["is_lightweight"]:
        lat_shap_img = get_explain_file(cfg["explain_dir"], "shap_analysis", "mil_latent_dims_shap_summary.png")
        dims_genes_tsv = get_explain_file(cfg["explain_dir"], "shap_analysis", "nonzero_dims_to_genes.tsv")
        if os.path.exists(lat_shap_img) or os.path.exists(dims_genes_tsv):
            with st.expander("🧬 LinearMIL Latent Dimensions (128D) & Top Driving Genes"):
                if os.path.exists(lat_shap_img):
                    st.image(lat_shap_img, width="stretch", caption="Top 20 GAT Latent Dimensions SHAP Summary")
                df_dims_genes = load_tsv(dims_genes_tsv)
                if df_dims_genes is not None:
                    st.dataframe(df_dims_genes, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("📋 Inspect & Download Gene Ranking Tables (TSV)")
    st.space("small")

    tsv_subfolders = {
        get_all_shap_tsv_filename(cfg["prefix"]): "shap_analysis",
        get_gene_attention_map_tsv_filename(cfg["prefix"]): "gene_saliency_attention",
        **{get_gene_shap_tsv_filename(cfg["prefix"], feat): "shap_analysis" for feat in active_node_features},
    }
    if cfg["is_lightweight"]:
        tsv_subfolders["mil_latent_dims_shap.tsv"] = "shap_analysis"

    tsv_file_choice = st.selectbox("Select Ranking Table to Inspect:", list(tsv_subfolders.keys()))
    target_tsv_path = get_explain_file(cfg["explain_dir"], tsv_subfolders[tsv_file_choice], tsv_file_choice)
    df_tsv = load_tsv(target_tsv_path)

    if df_tsv is not None:
        gene_col = next((c for c in ("gene_name", "feature", "gene_id") if c in df_tsv.columns), None)
        if shap_genes and gene_col:
            df_display = df_tsv[df_tsv[gene_col].astype(str).str.upper().isin(shap_genes)]
            st.dataframe(df_display, width="stretch")
            st.caption(f"Showing {len(df_display)} of {len(df_tsv)} genes matching the selected gene list.")
        else:
            df_display = df_tsv
            st.dataframe(df_display, width="stretch")

        st.download_button(
            label="💾 Download Table (TSV)",
            data=df_display.to_csv(sep="\t", index=False),
            file_name=tsv_file_choice,
            mime="text/tab-separated-values",
        )
    else:
        show_path_alert("warning", f'Selected table "{tsv_file_choice}" not found at:', target_tsv_path)


# =============================================================================
# TAB 3: INTERACTIVE PPI GRAPH, GENE SALIENCY & MOLECULAR PROFILES
# =============================================================================
with tab_graph:
    st.header("🕸️ Interactive PPI Network & Top Salient Gene Profiles")
    st.markdown("""
        Visualizes the protein-protein interactions and the **most salient genes** prioritized by the model during prediction.

        **How to explore the network:**
        - **Node Color & Size (Saliency Score):** Reflects each gene's learned importance (*Gradient × Input* saliency) for the model's prediction.
        - **Edge Thickness:** Thicker connections represent gene pairs weighted more strongly by the graph attention layers.
        - **Highlighted Target Genes (`🌟`):** Genes from your active pathway or search filter are outlined with a glowing red halo.
        - **In-Map Gene Search:** Use the search box in the top-right corner of the network canvas to locate and zoom into any gene.
    """)
    if st.button("▶️ Generate / Refresh Network & Gene Saliency Results", key="btn_run_graph"):
        run_explainability_step(selected_model, step_name="graph", cohort_key=selected_cohort_key, is_lightweight=cfg["is_lightweight"])

    graph_genes = render_gene_pathway_box("graph")

    edges_path = get_explain_file(cfg["explain_dir"], "gene_networks", get_edge_attention_tsv_filename(cfg["prefix"]))
    saliency_path = get_explain_file(
        cfg["explain_dir"], "gene_saliency_attention", get_gene_attention_map_tsv_filename(cfg["prefix"])
    )
    edges_df = load_tsv(edges_path)
    saliency_df = load_tsv(saliency_path)
    has_target_filter = bool(graph_genes)

    if edges_df is not None and saliency_df is None:
        show_path_alert(
            "warning",
            "⚠️ Gene saliency scores not found at:",
            saliency_path,
            "— nodes cannot be colored by Saliency Score until you click the refresh button above.",
        )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2.5, 2, 1])
    with col_ctrl1:
        if has_target_filter:
            st.slider(
                "Number of top interaction edges to display:",
                min_value=10, max_value=300, value=100, step=10,
                disabled=True,
                help="Disabled while a pathway or gene search filter is active.",
            )
            max_neighbors_per_node = st.number_input(
                "Max interacting partners per target gene:",
                min_value=1, max_value=50, value=5, step=1,
                help="Number of top-weighted interacting neighbor genes shown around each selected gene.",
            )
            top_n_edges = 100
        else:
            top_n_edges = st.slider(
                "Number of top interaction edges to display:",
                min_value=10, max_value=300, value=100, step=10,
            )
            max_neighbors_per_node = 5

    with col_ctrl2:
        color_strategy = st.selectbox(
            "Node Coloring Strategy:",
            ["Saliency Score", "Node Degree (Number of Connections)", "Connected Component (Clusters)"],
        )

    with col_ctrl3:
        enable_physics = st.checkbox("Enable Dynamic Physics", value=False)

    if edges_df is not None:
        graph_out_dir = os.path.join(cfg["explain_dir"], "gene_networks")
        html_graph = build_optimized_pyvis_subgraph(
            df_edges=edges_df,
            target_genes_tuple=tuple(graph_genes) if has_target_filter else None,
            top_n=top_n_edges,
            max_neighbors=max_neighbors_per_node,
            saliency_df=saliency_df,
            color_mode=color_strategy,
            enable_physics=enable_physics,
            out_dir=graph_out_dir,
        )
        if html_graph:
            st.iframe(html_graph, width="stretch", height=670)
        else:
            st.error("Error generating interactive PPI network.")
    else:
        show_path_alert("warning", "❌ Edge attention table not found at:", edges_path, "- Click the button above to generate network and saliency files.")

    st.markdown("---")
    st.subheader("🔥 Top Salient Genes: Modality Attention Heatmap & Molecular Profiles")
    st.markdown("""
        The plots below focus specifically on the **most salient genes identified by the model**—that is, the genes whose
        molecular features exert the strongest influence on the model's predicted outcome (ranked by *Gradient × Input* saliency score):

        1. **Gene × Molecular Modality Importance Heatmap (Top 25 Salient Genes):**
           Ranks the **top 25 most salient genes** from top to bottom and breaks down *which specific molecular modality*
           (e.g., gene expression vs. copy number variation) drives the model's attention for each gene. Higher values indicate stronger reliance.
        2. **Molecular Measurement Distributions Across Outcome Groups (Top 5 Salient Genes):**
           Displays the **actual measured molecular values** (such as normalized expression or copy number) across target outcome classes
           for the **top 5 most salient genes**, showing how their underlying biological measurements differ between classes.
    """)

    heatmap_img_path = get_explain_file(
        cfg["explain_dir"], "gene_saliency_attention", get_heatmap_png_filename(cfg["prefix"])
    )
    if os.path.exists(heatmap_img_path):
        st.image(heatmap_img_path, width="stretch", caption="Modality Importance Breakdown for the Top 25 Most Salient Genes")
    else:
        show_path_alert("warning", "Gene × Modality heatmap image not found at:", heatmap_img_path, "- Click the button above to generate it.")

    st.markdown("---")
    st.subheader("📊 Measured Molecular Distributions of the Top 5 Most Salient Genes")

    for feat in active_node_features:
        boxplot_img = get_explain_file(
            cfg["explain_dir"], "gene_saliency_attention", get_top5_boxplot_png_filename(cfg["prefix"], feat)
        )
        if os.path.exists(boxplot_img):
            st.image(boxplot_img, width="stretch")
            st.caption(f"Measured `{feat}` distribution across outcome classes for the top 5 most salient genes.")
        else:
            show_path_alert("info", f"Top 5 salient genes boxplot for `{feat}` not found at:", boxplot_img, "- Click the button above to generate boxplots.")


# =============================================================================
# TAB 4: PATHWAY ENRICHMENT
# =============================================================================
with tab_enrich:
    st.header("🧬 Functional Pathway Enrichment Analysis")
    st.markdown(
        "Tests whether genes prioritized by multi-omic SHAP attributions cluster into known biological "
        "pathways across MSigDB Hallmark, KEGG, Reactome, and Gene Ontology."
    )
    if st.button("▶️ Generate / Refresh Results for This Tab", key="btn_run_enrich"):
        run_explainability_step(
            selected_model,
            step_name="enrichment",
            cohort_key=selected_cohort_key,
            is_lightweight=cfg["is_lightweight"],
        )

    df_overall_raw = load_overall_enrichment_df(cfg["explain_dir"])
    df_res_raw = load_tsv(get_explain_file(cfg["explain_dir"], "pathway_enrichment", "ora_resistance_pathways.tsv"))
    df_sens_raw = load_tsv(get_explain_file(cfg["explain_dir"], "pathway_enrichment", "ora_sensitivity_pathways.tsv"))

    if df_overall_raw is None and df_res_raw is None and df_sens_raw is None:
        show_path_alert(
            "warning",
            "Pathway enrichment results not found in:",
            os.path.join(cfg["explain_dir"], "pathway_enrichment"),
            "— Click the button above to run enrichment analysis.",
        )
    else:
        selected_lib = st.selectbox(
            "Gene Set Library:",
            options=ENRICHMENT_LIBRARIES,
            index=0,
            key="enrich_library_select",
            help=(
                "MSigDB_Hallmark_2020: 50 concise, non-redundant cellular states (recommended starting point). "
                "KEGG_2021_Human & Reactome_2022: detailed signaling and metabolic cascades. "
                "GO_Biological_Process_2023: broad functional annotations."
            ),
        )
        st.caption(
            f"ℹ️ Only statistically significant pathways (**Adjusted P-value < {ENRICHMENT_FDR_CUTOFF}**, Benjamini-Hochberg FDR) "
            "are displayed below. Use the **Download Full TSV** buttons in each section to inspect borderline or non-significant terms."
        )

        df_res_lib = filter_enrichment_by_library(df_res_raw, selected_lib)
        df_sens_lib = filter_enrichment_by_library(df_sens_raw, selected_lib)
        df_overall_lib = filter_enrichment_by_library(df_overall_raw, selected_lib)

        if not df_res_lib.empty or not df_sens_lib.empty:
            st.markdown("---")
            st.subheader(
                "⚖️ Directional Enrichment: Pro-Resistance vs. Pro-Sensitivity",
                help=(
                    "Splits top model-prioritized genes by signed SHAP direction before running Over-Representation Analysis (ORA): "
                    "positive SHAP genes drive predictions toward resistance, whereas negative SHAP genes drive predictions toward sensitivity."
                ),
            )
            col_res, col_sens = st.columns(2)

            with col_res:
                st.markdown("#### 🔴 Pro-Resistance Drivers (`SHAP > 0`)")
                df_res_sig = (
                    df_res_lib[df_res_lib["Adjusted P-value"] < ENRICHMENT_FDR_CUTOFF]
                    if "Adjusted P-value" in df_res_lib.columns
                    else df_res_lib
                )
                if not df_res_sig.empty:
                    top_r = df_res_sig.iloc[0]
                    st.metric(
                        "Top Resistance Pathway",
                        str(top_r["Term"])[:38] + ("..." if len(str(top_r["Term"])) > 38 else ""),
                        delta=f"Adj. P = {top_r['Adjusted P-value']:.2e}" if "Adjusted P-value" in top_r else None,
                        delta_color="inverse",
                    )
                render_enrichment_panel(
                    df_res_lib,
                    title=f"Pro-Resistance ({selected_lib})",
                    bar_color="#dc2626",
                    download_filename=f"ora_resistance_{selected_lib}.tsv",
                )

            with col_sens:
                st.markdown("#### 🔵 Pro-Sensitivity Drivers (`SHAP < 0`)")
                df_sens_sig = (
                    df_sens_lib[df_sens_lib["Adjusted P-value"] < ENRICHMENT_FDR_CUTOFF]
                    if "Adjusted P-value" in df_sens_lib.columns
                    else df_sens_lib
                )
                if not df_sens_sig.empty:
                    top_s = df_sens_sig.iloc[0]
                    st.metric(
                        "Top Sensitivity Pathway",
                        str(top_s["Term"])[:38] + ("..." if len(str(top_s["Term"])) > 38 else ""),
                        delta=f"Adj. P = {top_s['Adjusted P-value']:.2e}" if "Adjusted P-value" in top_s else None,
                        delta_color="normal",
                    )
                render_enrichment_panel(
                    df_sens_lib,
                    title=f"Pro-Sensitivity ({selected_lib})",
                    bar_color="#0284c7",
                    download_filename=f"ora_sensitivity_{selected_lib}.tsv",
                )

        st.markdown("---")
        st.subheader(
            "📊 Overall Pathway Enrichment (Magnitude-Ranked Top Genes)",
            help=(
                "Runs Over-Representation Analysis (ORA) on the top 150 active genes ranked by total absolute multi-omic SHAP importance, "
                "capturing dominant biological themes regardless of effect direction."
            ),
        )
        render_enrichment_panel(
            df_overall_lib,
            title=f"Overall Top Genes ({selected_lib})",
            bar_color="#7c3aed",
            download_filename=f"ora_overall_{selected_lib}.tsv",
            top_n=10,
        )

        inspect_candidates = []
        for tag, df_src in [
            ("Resistance", df_res_lib),
            ("Sensitivity", df_sens_lib),
            ("Overall", df_overall_lib),
        ]:
            if df_src is not None and not df_src.empty and {"Term", "Genes", "Adjusted P-value"}.issubset(df_src.columns):
                df_src_sig = df_src[df_src["Adjusted P-value"] < ENRICHMENT_FDR_CUTOFF]
                for _, row in df_src_sig.head(15).iterrows():
                    label = f"[{tag}] {row['Term']} (Adj. P = {row['Adjusted P-value']:.2e})"
                    inspect_candidates.append((label, str(row["Term"]), str(row["Genes"]), str(row.get("Overlap", "N/A"))))

        if inspect_candidates:
            with st.expander("🔍 Inspect Pathway Overlap Genes & Save for SHAP / PPI Exploration", expanded=False):
                choice_map = {item[0]: item for item in inspect_candidates}
                chosen_label = st.selectbox(
                    "Select a Significant Pathway to Inspect Overlapping Genes:",
                    options=list(choice_map.keys()),
                    key="enrich_inspect_term_select",
                )
                _, raw_term, raw_genes, overlap_str = choice_map[chosen_label]
                gene_list = [g.strip().upper() for g in raw_genes.split(";") if g.strip()]

                st.markdown(f"**Overlap Ratio:** `{overlap_str}` | **Matched Model Genes ({len(gene_list)}):**")
                st.code(", ".join(gene_list))

                if st.button("💾 Save Pathway Genes to Custom Gene Subsets", key="btn_save_enriched_subset"):
                    raw_data = load_user_subsets()
                    subset_key = f"Enriched: {raw_term[:45]}"
                    raw_data[subset_key] = gene_list
                    if save_user_subsets(raw_data):
                        st.success(
                            f"Saved '{subset_key}' ({len(gene_list)} genes)! You can now select it inside "
                            "'Change Displayed Genes' in the SHAP and PPI Network tabs."
                        )


# =============================================================================
# TAB 5 (MIL-ONLY): TISSUE ATTENTION DISTRIBUTION
# =============================================================================
if cfg["prefix"] == "mil":
    with tab_tissue_attn:
        st.header("🧪 Tissue Sample Weighting Across Anatomical Sites (MIL Attention)")
        st.markdown("""
            Shows how much weight the patient-level model assigns to each tissue type when predicting patient outcome.

            **Why Multi-Tissue Patients Matter:**
            For each patient, tissue weights sum to 100% ($1.0$). When a patient has only a single tissue sample ($N=1$),
            that sample automatically receives 100% weight. Filtering for **Multi-Tissue Mixed Bags** restricts the summary
            to patients with multiple distinct tissues, revealing which tissue type the model genuinely prioritizes.
        """)
        if st.button("▶️ Generate / Refresh Results for This Tab", key="btn_run_tissue_attn"):
            run_explainability_step(
                selected_model,
                step_name="tissue_attn",
                cohort_key=selected_cohort_key,
                is_lightweight=cfg["is_lightweight"],
            )
        st.markdown("---")

        tissue_tsv_path = get_explain_file(
            cfg["explain_dir"], "feature_importance", FN_MIL_TISSUE_ATTENTION_AGGREGATION
        )
        df_tissue_att = load_tsv(tissue_tsv_path)

        if df_tissue_att is not None and not df_tissue_att.empty:
            cohort_filter = st.radio(
                "Select Patient Cohort Filter for Analysis:",
                ["Multi-Tissue Mixed Bags (Unbiased Competition, N > 1)", "All Patients (Includes Single-Sample Bags)"],
                horizontal=True,
            )
            is_mixed = "Unbiased" in cohort_filter and "mean_attention_mixed" in df_tissue_att.columns
            mean_col = "mean_attention_mixed" if is_mixed else "mean_attention_all"
            pct_col = "pct_attention_mixed" if is_mixed else "pct_attention_all"

            df_sorted = df_tissue_att.sort_values(by=mean_col, ascending=False).reset_index(drop=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Attended Tissue", str(df_sorted.iloc[0]["tissue_type"]))
            col2.metric("Mean Attention Weight", f"{df_sorted.iloc[0][mean_col]:.4f}")
            col3.metric("Relative Attention Share", f"{df_sorted.iloc[0][pct_col]:.1f}%")

            st.markdown("---")
            st.subheader(
                f"📊 Mean Sample Weight by Tissue ({'Multi-Tissue Patients Only' if is_mixed else 'All Patients'})"
            )

            tissues = df_sorted["tissue_type"].astype(str).tolist()
            mean_vals = pd.to_numeric(df_sorted[mean_col], errors="coerce").fillna(0.0).values
            pct_vals = pd.to_numeric(df_sorted[pct_col], errors="coerce").fillna(0.0).values

            fig, ax = plt.subplots(figsize=(9.5, 4.3))
            x_pos = np.arange(len(tissues))
            bars = ax.bar(
                x_pos,
                mean_vals,
                width=0.56,
                color="#7c3aed",
                edgecolor="#5b21b6",
                linewidth=1.0,
                alpha=0.90,
            )

            max_val = float(np.max(mean_vals)) if len(mean_vals) > 0 else 1.0
            y_offset = max(max_val * 0.025, 0.003)
            for bar, pct in zip(bars, pct_vals):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + y_offset,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#1e1b4b",
                )

            rotate_labels = len(tissues) > 4 or any(len(t) > 12 for t in tissues)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                tissues,
                rotation=25 if rotate_labels else 0,
                ha="right" if rotate_labels else "center",
                fontsize=9.5,
            )
            ax.set_ylabel("Mean Sample Weight", fontsize=10, fontweight="bold", color="#1e1b4b")
            ax.set_xlabel("Tissue Type", fontsize=10, fontweight="bold", color="#1e1b4b")
            ax.set_title(
                "Mean Sample Attention Weight (with Relative Share % Annotated)",
                fontsize=11,
                fontweight="bold",
                color="#1e1b4b",
                pad=14,
            )
            ax.set_ylim(0, max(max_val * 1.18, 0.05))
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            st.markdown("---")
            st.subheader("📋 Tissue Summary Table")
            st.dataframe(df_tissue_att, width="stretch")

            raw_patient_tsv = get_explain_file(
                cfg["explain_dir"], "feature_importance", "mil_patient_tissue_attention_raw.tsv"
            )
            df_raw_patient = load_tsv(raw_patient_tsv)
            if df_raw_patient is not None:
                with st.expander("🔍 Inspect Per-Sample Attention Weights by Patient"):
                    st.caption(
                        "Displays full sample IDs and per-sample attention weights alongside total samples and distinct tissues per patient."
                    )
                    if "sample_id" not in df_raw_patient.columns:
                        st.info(
                            "Column `sample_id` not yet present in this cached file — click **▶️ Generate / Refresh Results for This Tab** above to populate full sample IDs."
                        )

                    patient_options = ["All Patients"] + sorted(
                        df_raw_patient["patient_id"].astype(str).unique().tolist()
                    )
                    selected_pid_filter = st.selectbox(
                        "Filter by Patient ID:",
                        options=patient_options,
                        index=0,
                        key="mil_raw_patient_filter",
                    )

                    df_raw_view = (
                        df_raw_patient
                        if selected_pid_filter == "All Patients"
                        else df_raw_patient[df_raw_patient["patient_id"].astype(str) == selected_pid_filter]
                    )
                    st.dataframe(df_raw_view, width="stretch", hide_index=True)

                    st.download_button(
                        label="💾 Download Per-Sample Attention Weights (TSV)",
                        data=df_raw_patient.to_csv(sep="\t", index=False),
                        file_name="mil_patient_tissue_attention_raw.tsv",
                        mime="text/tab-separated-values",
                    )
        else:
            show_path_alert(
                "warning",
                "❌ Tissue attention table not found at:",
                tissue_tsv_path,
                "- Click the button above to generate tissue attention outputs.",
            )


# =============================================================================
# TAB 6: SAMPLE & PATIENT SIMILARITY MAPS (PCA / UMAP)
# =============================================================================
with tab_pca:
    st.header("📉 Sample & Patient Similarity Maps (PCA / UMAP)")
    if cfg["prefix"] == "gat":
        st.markdown("""
            Projects the molecular profiles learned by the tissue-sample model (GAT) onto 2D similarity maps (PCA and UMAP).

            **What you can observe:**
            - **Tissue Grouping:** Whether samples naturally cluster by tissue type.
            - **Clinical Outcome Separation:** Whether tissue profiles separate according to patient clinical status.
            - **Within-Patient Similarity:** Whether multiple tissue samples from the same patient group close together.
        """)
    else:
        st.markdown("""
            Projects the molecular profiles learned by the patient-level model (MIL) onto 2D similarity maps (PCA and UMAP).

            **What you can observe:**
            1. **Patient-Level Profiles:** Overall molecular summary per patient combining all available tissue samples.
            2. **Tissue-Sample Profiles:** Individual tissue representations refined during patient-level training.
        """)

    if st.button("▶️ Generate / Refresh Results for This Tab", key="btn_run_pca"):
        run_explainability_step(selected_model, step_name="pca", cohort_key=selected_cohort_key, is_lightweight=cfg["is_lightweight"])
    st.markdown("---")
    st.space("small")

    if cfg["is_lightweight"]:
        suf = st.session_state.get("lw_eval_cohort", "oof")
        cohort_choice = "Out-Of-Fold Validation Set (OOF)" if suf == "oof" else "Held-Out Test Set (TEST)"
        st.info(f"Showing PCA / UMAP projections for active sidebar evaluation subset: **{cohort_choice}**")
    else:
        cohort_choice = st.radio(
            "Select Dataset Cohort for PCA / UMAP Visualization:",
            ["Full Dataset (All Patients & Samples)", "Test Set Only"],
            horizontal=True,
        )
        suf = "all" if "Full" in cohort_choice else "test"

    def resolve_pca_image(base_prefix, fallback_const):
        img = get_explain_file(cfg["explain_dir"], "embeddings_pca", f"{base_prefix}_{suf}.png")
        return img if os.path.exists(img) else get_explain_file(cfg["explain_dir"], "embeddings_pca", fallback_const)

    if cfg["prefix"] == "gat":
        gat_pca_img = resolve_pca_image("gat_sample_embeddings_pca", FN_GAT_SAMPLE_PCA_PNG)
        if os.path.exists(gat_pca_img):
            st.image(gat_pca_img, width="stretch", caption=f"Tissue Sample Similarity Map ({cohort_choice})")
        else:
            show_path_alert("warning", "❌ GAT sample PCA image not found at:", gat_pca_img, "- Click the button above to generate PCA plots.")
    else:
        mil_patient_pca_img = resolve_pca_image("mil_patient_embeddings_pca", FN_MIL_PATIENT_PCA_PNG)
        mil_sample_pca_img = resolve_pca_image("mil_sample_embeddings_pca", FN_MIL_SAMPLE_PCA_PNG)

        st.subheader(f"1. Patient-Level Profiles ({suf.upper()})")
        if os.path.exists(mil_patient_pca_img):
            st.image(mil_patient_pca_img, width="stretch", caption=f"Patient-Level Similarity Map ({cohort_choice})")
        else:
            show_path_alert("warning", "❌ Patient PCA image not found at:", mil_patient_pca_img)

        st.markdown("---")

        st.subheader(f"2. Refined Tissue-Sample Profiles ({suf.upper()})")
        if os.path.exists(mil_sample_pca_img):
            st.image(mil_sample_pca_img, width="stretch", caption=f"Tissue-Sample Similarity Map ({cohort_choice})")
        else:
            show_path_alert("warning", "❌ MIL sample PCA image not found at:", mil_sample_pca_img)