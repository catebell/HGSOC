import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config_file_tree
from utils_explainability import (
    to_rel_path,
    FN_USER_GENE_SUBSETS_JSON,
    FN_KEGG_BARPLOT_PNG,
    FN_KEGG_TOP_PATHWAYS_TSV,
    get_performance_report_filenames,
    get_gene_shap_tsv_filename,
    get_gene_shap_png_filename,
    get_top5_boxplot_png_filename,
    get_importance_tsv_filename,
    get_edge_attention_tsv_filename,
    get_gene_attention_map_tsv_filename,
    get_all_shap_tsv_filename,
    get_all_shap_png_filename,
    get_heatmap_png_filename, FN_MIL_TISSUE_ATTENTION_AGGREGATION
)

USER_SUBSETS_FILE = os.path.join(config_file_tree.ROOT_DIR, FN_USER_GENE_SUBSETS_JSON)

# STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="GNN & MIL BioExplainability Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLING CUSTOMIZATION: Increased normal text font size while restoring large st.metric card styling
st.markdown("""
    <style>
    /* Body text, markdown paragraphs, and list items */
    .stMarkdown p, .stMarkdown li, div[data-testid="stMarkdownContainer"] p {
        font-size: 1.12rem !important;
    }
    /* Streamlit alert boxes (info, success, warning, error) */
    .stAlert div[role="alert"] p, div[data-testid="stAlert"] p {
        font-size: 1.08rem !important;
    }
    /* Captions and small text */
    .stCaption, div[data-testid="stCaptionContainer"] p, caption {
        font-size: 0.98rem !important;
    }
    /* Form labels, input fields, selectbox, radio, and slider text */
    label, .stSelectbox label, .stTextInput label, .stRadio label, .stSlider label {
        font-size: 1.08rem !important;
    }
    /* Dataframes and table cells */
    .stDataFrame, div[data-testid="stTable"] {
        font-size: 1.05rem !important;
    }

    /* RE-STYLED METRIC CARDS (TAB 1 SUMMARY) */
    div[data-testid="stMetric"] {
        background-color: rgba(240, 242, 246, 0.6);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
    }
    div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] p {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #1a202c !important;
    }
    </style>
""", unsafe_allow_html=True)


# HELPERS FOR HOVER TOOLTIPS ON MODEL & DIRECTORY PATHS
def format_model_hover_path(model_path):
    """
    Returns HTML displaying only the filename of the .pt checkpoint,
    with a browser hover tooltip (title attribute) displaying the relative path.
    """
    if not model_path:
        return ""
    filename = os.path.basename(model_path)
    rel_path = to_rel_path(model_path)
    return f'<span title="{rel_path}" style="cursor: help; background-color: rgba(135, 131, 120, 0.18); padding: 2px 6px; border-radius: 4px; color: #1e293b;">{filename}</span>'


def format_dir_hover_path(dir_path):
    """
    Returns HTML displaying only the last two path components of the explainability dir,
    with a browser hover tooltip showing the full path starting from 6_explainability.
    """
    if not dir_path:
        return ""
    rel_path = to_rel_path(dir_path)
    norm_path = os.path.normpath(rel_path)
    parts = norm_path.split(os.sep)
    if len(parts) >= 2:
        display_str = f"{parts[-2]}/{parts[-1]}"
    else:
        display_str = norm_path
    return f'<span title="{rel_path}" style="cursor: help; background-color: rgba(135, 131, 120, 0.18); padding: 2px 6px; border-radius: 4px; color: #1e293b;">{display_str}</span>'


# Default Built-in Biological Gene Pathways
DEFAULT_GENE_GROUPS = {
    "Top Edges Attention | All Genes (Default)": [],
    "Protein Kinases": ["MAPK1", "MAPK3", "AKT1", "PIK3CA", "EGFR", "ERBB2", "CDK1", "CDK2"],
    "DNA Damage & Repair": ["TP53", "BRCA1", "BRCA2", "ATM", "ATR", "PARP1", "CHEK2"],
    "Lipid Metabolism": ["APOC3", "LPL", "FABP4", "ADIPOQ", "PNPLA3", "PPARA"],
    "Cell Adhesion & Angiogenesis": ["CDH5", "RAMP3", "WNT3A", "CALCRL", "COX7A1"]
}


# PERSISTENT USER SUBSETS MANAGER
def load_user_subsets():
    """Load custom user gene subsets from local JSON config file."""
    if os.path.exists(USER_SUBSETS_FILE):
        try:
            with open(USER_SUBSETS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_subsets(subsets_dict):
    """Save custom user gene subsets to local JSON config file."""
    try:
        with open(USER_SUBSETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(subsets_dict, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving gene subsets: {e}")
        return False


# FILE LOADING & MODEL DISCOVERY HELPERS
def load_tsv(filepath):
    """Safely loads TSV/CSV tabular data if the file exists."""
    if filepath and os.path.exists(filepath):
        try:
            return pd.read_csv(filepath, sep='\t')
        except Exception:
            try:
                return pd.read_csv(filepath)
            except Exception:
                return None
    return None


def find_available_models():
    """Scans trained_models directory for available PyTorch model checkpoints (.pt)."""
    base_dir = config_file_tree.TRAINED_MODELS_DIR
    gat_models = []
    mil_models = []

    if os.path.exists(base_dir):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.pt'):
                    full_path = os.path.join(root, file)
                    rel_path = to_rel_path(full_path)
                    if "GAT" in file.upper():
                        gat_models.append((rel_path, full_path))
                    elif "MIL" in file.upper():
                        mil_models.append((rel_path, full_path))

    return gat_models, mil_models


def get_model_config(selected_model_name):
    """
    Returns configuration parameters and directory paths dynamically calculated
    inside 6_explainability/models_explained/[relative_model_path].
    """
    if "GAT" in selected_model_name:
        default_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
        default_path = os.path.join(
            default_dir,
            f"best_GAT_{config_file_tree.SAMPLE_TARGET_COL}_{'withMeta' if config_file_tree.USE_SAMPLES_METADATA else 'noMeta'}.pt"
        )
        model_path = st.session_state.get('custom_gat_path', default_path)
        model_dir = os.path.dirname(model_path)
        rel_model_dir = os.path.relpath(model_dir, config_file_tree.TRAINED_MODELS_DIR)
        explain_dir = os.path.join(config_file_tree.EXPLAINABILITY_DIR, "models_explained", rel_model_dir)
        target_col = config_file_tree.SAMPLE_TARGET_COL
        use_extra = config_file_tree.USE_SAMPLES_METADATA
        extra_type = "Samples Metadata"
        prefix = "gat"
    else:
        default_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.PATIENT_ENCODER_FOLDER)
        default_path = os.path.join(
            default_dir,
            f"best_MIL_{config_file_tree.PATIENT_TARGET_COL}_{'withClin' if config_file_tree.USE_CLINICAL_DATA else 'noClin'}.pt"
        )
        model_path = st.session_state.get('custom_mil_path', default_path)
        model_dir = os.path.dirname(model_path)
        rel_model_dir = os.path.relpath(model_dir, config_file_tree.TRAINED_MODELS_DIR)
        explain_dir = os.path.join(config_file_tree.EXPLAINABILITY_DIR, "models_explained", rel_model_dir)
        target_col = config_file_tree.PATIENT_TARGET_COL
        use_extra = config_file_tree.USE_CLINICAL_DATA
        extra_type = "Patient Clinical Features"
        prefix = "mil"

    os.makedirs(explain_dir, exist_ok=True)

    return {
        "model_path": model_path,
        "model_dir": model_dir,
        "explain_dir": explain_dir,
        "target_col": target_col,
        "use_extra": use_extra,
        "extra_type": extra_type,
        "prefix": prefix,
        "task": config_file_tree.TASK
    }


def load_performance_report(model_dir, prefix):
    """Locates performance report in the active model directory."""
    possible_names = get_performance_report_filenames(prefix)
    for fname in possible_names:
        report_path = os.path.join(model_dir, fname)
        if os.path.exists(report_path):
            try:
                sep = '\t' if fname.endswith('.tsv') else ','
                return pd.read_csv(report_path, sep=sep), report_path
            except Exception:
                pass
    return None, os.path.join(model_dir, f"report_kfold_test_results_{prefix.upper()}.csv")


def parse_report_metrics(df):
    """Extracts mean and std/var metric values from the report dataframe."""
    if df is None or df.empty:
        return None, {}

    if 'fold' in df.columns:
        fold_col_str = df['fold'].astype(str).str.strip().str.lower()
        summary_labels = {'mean', 'std', 'var', 'variance', 'mean ± var', 'mean +- var'}
        df_folds = df[~fold_col_str.isin(summary_labels)].copy()
    else:
        df_folds = df.copy()

    metrics_summary = {}
    metric_cols = [c for c in df.columns if c != 'fold']

    for col in metric_cols:
        numeric_vals = pd.to_numeric(df_folds[col], errors='coerce').dropna()
        if not numeric_vals.empty:
            mean_val = float(numeric_vals.mean())
            var_val = float(numeric_vals.var()) if len(numeric_vals) > 1 else 0.0
            std_val = float(numeric_vals.std()) if len(numeric_vals) > 1 else 0.0
            metrics_summary[col] = {
                'mean': mean_val,
                'var': var_val,
                'std': std_val,
                'display': f"{mean_val:.4f} ± {var_val:.4f}"
            }

    return df_folds, metrics_summary


@st.cache_data
def build_optimized_pyvis_subgraph(
        df_edges,
        target_genes_tuple=None,
        top_n=100,
        saliency_df=None,
        color_mode="Saliency Score",
        enable_physics=False
):
    """Builds interactive PyVis network respecting target gene filters."""
    if df_edges is None or df_edges.empty:
        return None

    if target_genes_tuple and len(target_genes_tuple) > 0:
        target_set = {str(g).upper() for g in target_genes_tuple}
        filtered = df_edges[
            df_edges['gene_symbol_a'].astype(str).str.upper().isin(target_set) |
            df_edges['gene_symbol_b'].astype(str).str.upper().isin(target_set)
            ]
        df_sub = filtered.head(top_n) if not filtered.empty else df_edges.head(top_n)
    else:
        df_sub = df_edges.head(top_n)

    G = nx.Graph()
    saliency_dict = {}
    if saliency_df is not None and not saliency_df.empty:
        saliency_dict = dict(zip(saliency_df['gene_name'], saliency_df['saliency_score']))

    min_w = df_sub['attention_weight'].min()
    max_w = df_sub['attention_weight'].max()
    denom_w = (max_w - min_w) if max_w != min_w else 1.0

    for _, row in df_sub.iterrows():
        u = str(row['gene_symbol_a'])
        v = str(row['gene_symbol_b'])
        raw_w = float(row['attention_weight'])
        norm_width = 2.0 + 4.0 * ((raw_w - min_w) / denom_w)
        G.add_edge(
            u, v, weight=raw_w, width=norm_width,
            title=f"Interaction: {u} - {v}\nAttention Weight: {raw_w:.5f}",
            color={'color': 'rgba(80, 120, 170, 0.75)', 'highlight': '#e63946'}
        )

    pos = nx.spring_layout(G, k=0.18, seed=42)
    for node, (x, y) in pos.items():
        G.nodes[node]['x'] = float(x * 1000)
        G.nodes[node]['y'] = float(y * 1000)

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    min_deg = min(degrees.values()) if degrees else 0

    components_list = list(nx.connected_components(G))
    comp_map = {node: idx for idx, comp in enumerate(components_list) for node in comp}

    node_sal_scores = [saliency_dict.get(node, 1e-6) for node in G.nodes()]
    sal_logs = [np.log10(max(s, 1e-8)) for s in node_sal_scores]
    min_log_sal = min(sal_logs) if sal_logs else -8.0
    max_log_sal = max(sal_logs) if sal_logs else 0.0
    denom_log_sal = (max_log_sal - min_log_sal) if max_log_sal != min_log_sal else 1.0

    for node in G.nodes():
        deg = degrees[node]
        comp_idx = comp_map[node]
        sal_score = saliency_dict.get(node, 1e-6)
        norm_deg = (deg - min_deg) / (max_deg - min_deg) if max_deg != min_deg else 0.5

        if color_mode == "Node Degree (Number of Connections)":
            hex_color = mcolors.to_hex(plt.colormaps['plasma'](0.15 + 0.8 * norm_deg))
            color_title = f"Gene: {node}\nConnections (Degree): {deg}\nRaw Saliency Score: {sal_score:.6f}"
            node_size = 14 + 18 * norm_deg
        elif color_mode == "Connected Component (Clusters)":
            hex_color = mcolors.to_hex(plt.colormaps['Set3'](comp_idx % 12))
            color_title = f"Gene: {node}\nCluster Component ID: #{comp_idx + 1}\nConnections: {deg}"
            node_size = 16 + 14 * norm_deg
        else:
            log_val = np.log10(max(sal_score, 1e-8))
            norm_sal = (log_val - min_log_sal) / denom_log_sal if denom_log_sal > 0 else 0.5
            hex_color = mcolors.to_hex(plt.colormaps['Spectral_r'](norm_sal))
            color_title = f"Gene: {node}\nRaw Saliency Score: {sal_score:.6f}\nRelative Rank (Log Scale): {norm_sal * 100:.1f}%\nConnections: {deg}"
            node_size = 14 + 18 * float(norm_sal)

        G.nodes[node]['size'] = node_size
        G.nodes[node]['color'] = {'background': hex_color, 'border': '#333333',
                                  'highlight': {'background': '#A8DDA8', 'border': '#111111'}}
        G.nodes[node]['font'] = {'color': '#222222', 'size': 13, 'face': 'arial'}
        G.nodes[node]['title'] = color_title

    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=False)
    net.from_nx(G)
    if not enable_physics:
        net.toggle_physics(False)

    html_path = "temp_network.html"
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


# SIDEBAR: NAVIGATION & CONTROL PANEL
st.sidebar.title("🧬 BioExplainability Navigator")
st.sidebar.markdown("---")

# HELPER PER LA FORMATTAZIONE DELLE OPZIONI NEI MENU A TENDINA
def format_model_option(model_tuple):
    """
    Formatta l'etichetta del modello nel menu a tendina mostrando:
    nome_cartella_esperimento / MIL_o_GAT / nome_file.pt
    """
    full_path = model_tuple[1]
    parts = os.path.normpath(full_path).split(os.sep)
    if len(parts) >= 3:
        return f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
    elif len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return os.path.basename(full_path)


# =============================================================================
# 1. LOAD MODELS SECTION (SIDEBAR)
# =============================================================================
st.sidebar.subheader(
    "📂 Load Models",
    help="Select pre-trained PyTorch model checkpoints (.pt) from the trained_models/ directory for analysis.\n\nExecuting the pipeline computes post-hoc explainability outputs (edge attention, node feature saliency, permutation importances, and SHAP values) for the active architecture."
)

gat_list, mil_list = find_available_models()

# Calcolo degli indici predefiniti in base alle cartelle specificate in config_file_tree
default_gat_folder = getattr(config_file_tree, 'SAMPLE_ENCODER_FOLDER', '')
default_gat_idx = 0
for i, (rel_p, full_p) in enumerate(gat_list):
    if default_gat_folder and default_gat_folder in full_p:
        default_gat_idx = i
        break

default_mil_folder = getattr(config_file_tree, 'PATIENT_ENCODER_FOLDER', '')
default_mil_idx = 0
for i, (rel_p, full_p) in enumerate(mil_list):
    if default_mil_folder and default_mil_folder in full_p:
        default_mil_idx = i
        break

if gat_list:
    selected_gat = st.sidebar.selectbox(
        "GAT Model Checkpoint:",
        options=gat_list,
        index=default_gat_idx,
        format_func=format_model_option,
        help="Select a trained Graph Attention Network checkpoint evaluating individual tissue sample graphs."
    )
    st.session_state['custom_gat_path'] = selected_gat[1]
else:
    st.sidebar.warning("No GAT `.pt` models found.")

if mil_list:
    selected_mil = st.sidebar.selectbox(
        "Patient MIL Checkpoint:",
        options=mil_list,
        index=default_mil_idx,
        format_func=format_model_option,
        help="Select a trained Multiple Instance Learning checkpoint evaluating patient bags of tissue graphs."
    )
    st.session_state['custom_mil_path'] = selected_mil[1]
else:
    st.sidebar.warning("No MIL `.pt` models found.")

st.sidebar.markdown("---")

# 2. ARCHITECTURE SELECTION SECTION
st.sidebar.subheader("⚙️ Architecture Selection")
st.sidebar.caption(
    "Select the model level to display in the dashboard. The selected level dictates the data currently shown across all dashboard tabs.")

selected_model = st.sidebar.radio(
    "Select Active Model Level:",
    ["GAT (Tissue Sample Level)", "Patient MIL (Patient Level)"]
)

cfg = get_model_config(selected_model)

run_pipeline_clicked = st.sidebar.button(
    "🚀 Run Explainability Pipeline",
    type="primary",
    help="Click to execute the post-hoc explainability pipeline for the selected model."
)

st.sidebar.markdown("---")

# 3. ACTIVE FILE TRACE SECTION
st.sidebar.subheader("📂 Active File Trace")
st.sidebar.markdown(
    f"""
    <div style="background-color: rgba(28, 131, 225, 0.08); padding: 12px; border-radius: 8px; border-left: 4px solid #1c83e1; line-height: 1.7;">
        <strong>Task:</strong> <code>{cfg['task']}</code><br>
        <strong>Target Column:</strong> <code>{cfg['target_col']}</code><br>
        <strong>Model Checkpoint:</strong> {format_model_hover_path(cfg['model_path'])}<br>
        <strong>Explainability Dir:</strong> {format_dir_hover_path(cfg['explain_dir'])}
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# 4. GENE & PATHWAY SEARCH & EDITOR SECTION
st.sidebar.subheader("🧬 Gene & Pathway Search & Editor")

user_subsets = load_user_subsets()
combined_groups = dict(DEFAULT_GENE_GROUPS)
combined_groups.update(user_subsets)

selected_group = st.sidebar.selectbox(
    "Select Biological Group / Pathway:",
    options=list(combined_groups.keys())
)

preset_genes = combined_groups.get(selected_group, [])
if preset_genes:
    st.sidebar.caption(
        f"**Included Genes ({len(preset_genes)}):** `{', '.join(preset_genes[:8])}{'...' if len(preset_genes) > 8 else ''}`")

# EXPANDER TO EDIT OR CREATE PATHWAYS IN SIDEBAR
with st.sidebar.expander("Edit Selected Pathway"):
    st.markdown("#### Modify Selected Pathway")
    edit_genes_text = st.text_area(
        f"Edit Included Genes for '{selected_group}':",
        value=", ".join(preset_genes),
        help="Modify comma-separated genes and click save to update this pathway in user config."
    )
    if st.button("💾 Save Pathway Changes"):
        updated_gene_list = [g.strip().upper() for g in edit_genes_text.split(",") if g.strip()]
        user_subsets[selected_group] = updated_gene_list
        if save_user_subsets(user_subsets):
            st.success(f"Updated pathway '{selected_group}'!")
            st.rerun()

    with st.sidebar.expander("Manage Custom Subsets"):
        st.markdown("#### Create New Pathway")
        new_subset_name = st.text_input("New Pathway Name:", placeholder="e.g. Immune Genes")
        new_subset_genes = st.text_input("New Pathway Genes:", placeholder="e.g. TP53, BRCA1, EGFR")
        if st.button("➕ Create New Pathway"):
            if new_subset_name.strip() and new_subset_genes.strip():
                gene_list = [g.strip().upper() for g in new_subset_genes.split(",") if g.strip()]
                user_subsets[new_subset_name.strip()] = gene_list
                if save_user_subsets(user_subsets):
                    st.success(f"Created pathway '{new_subset_name}'!")
                    st.rerun()

custom_gene_search = st.sidebar.text_input(
    "Or type specific Gene Symbols (comma separated):",
    placeholder="e.g. TP53, APOC3, BRCA1"
)

user_genes = []
if custom_gene_search.strip():
    user_genes = [g.strip().upper() for g in custom_gene_search.split(",") if g.strip()]
elif preset_genes:
    user_genes = preset_genes

apply_filter_clicked = st.sidebar.button("🔄 Apply Gene Filter & Update Plots", type="primary")

# MAIN DASHBOARD AREA
st.title(f"📊 Explainability Dashboard — {selected_model}")
st.markdown(
    f"Active Checkpoint: {format_model_hover_path(cfg['model_path'])} | Output Directory: {format_dir_hover_path(cfg['explain_dir'])}",
    unsafe_allow_html=True
)
st.space("small")

# HANDLE EXPLAINABILITY PIPELINE EXECUTION (TRIGGERED FROM SIDEBAR)
if run_pipeline_clicked:
    target_script = "explainability_gat_samples.py" if "GAT" in selected_model else "explainability_mil_patients.py"
    script_path = os.path.join(config_file_tree.EXPLAINABILITY_DIR, target_script)

    if not os.path.exists(script_path):
        script_path = os.path.join(config_file_tree.ROOT_DIR, target_script)

    st.info(f"Executing script `{to_rel_path(script_path)}` using Python environment...")

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.text("⏳ Starting explainability pipeline...")

    env = os.environ.copy()
    env["PYTHONPATH"] = config_file_tree.ROOT_DIR + os.pathsep + env.get("PYTHONPATH", "")

    step_triggers = [
        ("Starting", 0.10, "Loading datasets and model checkpoints..."),
        ("Edge Attention", 0.30, "Extracting attention weights from PPI network edges..."),
        ("Attention Map", 0.50, "Computing Gene x Feature attention map..."),
        ("importance", 0.65, "Calculating permutation feature importances..."),
        ("SHAP", 0.80, "Extracting SHAP values..."),
        ("visual plots", 0.90, "Generating visual network plots and boxplots..."),
        ("Complete", 1.00, "Completed!")
    ]

    process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=config_file_tree.ROOT_DIR,
        env=env,
        bufsize=1
    )

    stdout_logs = []
    current_progress = 0.0

    for line in iter(process.stdout.readline, ''):
        stdout_logs.append(line)
        line_str = line.strip()

        for keyword, pct, status_msg in step_triggers:
            if keyword.lower() in line_str.lower() and pct > current_progress:
                current_progress = pct
                progress_bar.progress(current_progress)
                status_text.text(f"🔄 {status_msg}")
                break

    process.stdout.close()
    return_code = process.wait()

    if return_code == 0:
        progress_bar.progress(1.0)
        status_text.text("✅ Explainability analysis completed successfully!")
        st.success("Explainability pipeline completed successfully!")
        with st.expander("Show Execution Logs"):
            st.code("".join(stdout_logs))
    else:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error running script `{target_script}` at path: `{to_rel_path(script_path)}` (`{script_path}`)")
        with st.expander("Show Execution Logs"):
            st.code("".join(stdout_logs))

# MAIN DASHBOARD TABS
if cfg['prefix'] == 'mil':
    tab_eval, tab_shap, tab_graph, tab_boxplots, tab_enrich, tab_tissue_attn = st.tabs([
        "📈 Evaluation & Performance Metrics",
        "🎯 Feature Importance & SHAP",
        "🕸️ Interactive PPI Graph & Attention",
        "📊 Gene Feature Boxplots",
        "🧬 Functional Pathway Enrichment",
        "🧪 MIL Tissue Attention Distribution"
    ])
else:
    tab_eval, tab_shap, tab_graph, tab_boxplots, tab_enrich = st.tabs([
        "📈 Evaluation & Performance Metrics",
        "🎯 Feature Importance & SHAP",
        "🕸️ Interactive PPI Graph & Attention",
        "📊 Gene Feature Boxplots",
        "🧬 Functional Pathway Enrichment"
    ])

# =============================================================================
# TAB 1: EVALUATION & PERFORMANCE METRICS
# =============================================================================
with tab_eval:
    st.header("Model Performance & K-Fold Validation Results")
    st.markdown("""
        This section displays out-of-fold evaluation metrics across 5-fold cross-validation splits. 
        The summary cards show global mean values and variances across folds, providing a measure of model accuracy, generalization, and stability.
        """)

    report_df, report_file_path = load_performance_report(cfg['model_dir'], cfg['prefix'])
    df_clean_folds, summary_metrics = parse_report_metrics(report_df)

    if report_df is not None:
        st.success(f"Loaded performance metrics report from: `{to_rel_path(report_file_path)}`")

        col1, col2, col3, col4 = st.columns(4)
        if 'auc' in summary_metrics:
            col1.metric("Mean AUC-ROC ± Var", summary_metrics['auc']['display'])
        elif 'c_index' in summary_metrics:
            col1.metric("Mean C-Index ± Var", summary_metrics['c_index']['display'])

        if 'f1' in summary_metrics:
            col2.metric("Mean Macro F1 ± Var", summary_metrics['f1']['display'])

        if 'acc' in summary_metrics:
            col3.metric("Mean Accuracy ± Var", summary_metrics['acc']['display'])

        if 'test_loss' in summary_metrics:
            col4.metric("Mean Test Loss ± Var", summary_metrics['test_loss']['display'])

        st.markdown("---")
        st.subheader("📋 K-Fold Cross-Validation Breakdown Table")
        st.markdown("""
                **What you can observe:**
                Inspect performance metrics individually for each validation fold to verify cross-validation stability and detect potential variance or outliers across data splits.
                """)
        st.dataframe(report_df, use_container_width=True)

    else:
        st.error(
            f"❌ Performance report file not found at searched location: `{to_rel_path(report_file_path)}` (`{report_file_path}`)")

    st.markdown("---")
    st.subheader(f"Visualizing Feature Importances ({cfg['extra_type']})")
    st.markdown(f"""
        **What you can observe:**
        Permutation importance measures performance drop when values of sample metadata or patient clinical attributes are randomly shuffled. 
        Higher positive values indicate variables that strongly contribute to correct predictions.
        """)

    meta_filename = get_importance_tsv_filename(cfg['prefix'])
    meta_filepath = os.path.join(cfg['explain_dir'], meta_filename)
    if os.path.exists(meta_filepath):
        st.dataframe(pd.read_csv(meta_filepath, sep='\t'), use_container_width=True)
    else:
        st.warning(
            f"Feature importance data file not found at searched location: `{to_rel_path(meta_filepath)}` (`{meta_filepath}`)")

# =============================================================================
# TAB 2: SHAP & FEATURE IMPORTANCE
# =============================================================================
with tab_shap:
    st.header("SHAP Value Explanations Across All Node Features")
    st.markdown("""
        SHAP (Shapley Additive exPlanations) attributes model predictions to individual node features based on cooperative game theory.
        This tab highlights feature attribution beeswarm plots and detailed tabular rankings.
        """)
    if user_genes:
        st.success(
            f"Active Gene Filter ({len(user_genes)} target genes): {', '.join(user_genes[:10])}{'...' if len(user_genes) > 10 else ''}")

    st.subheader("🎯 Node Feature SHAP Summaries")
    st.markdown("""
        **What you can observe:**
        - **Horizontal Axis (SHAP Value):** Indicates impact on prediction score (positive values drive the model toward positive predictions).
        - **Color Gradient (Feature Value):** Represents high (red/purple) versus low (blue) raw input feature values.
        - **Vertical Distribution:** Genes are ranked top-to-bottom by global importance.
        """)
    st.space("small")

    node_feature_indices = getattr(config_file_tree, 'NODE_FEATURE_INDEX', [0, 1])
    node_feature_names = getattr(config_file_tree, 'NODE_FEATURE_NAMES', ['gene_expression', 'log2CopyNumber'])
    active_node_features = [node_feature_names[i] for i in node_feature_indices if i < len(node_feature_names)]

    shap_cols = st.columns(len(active_node_features)) if active_node_features else [st.container()]

    for idx, feat_name in enumerate(active_node_features):
        feat_shap_filename = get_gene_shap_png_filename(cfg['prefix'], feat_name)
        feat_shap_img = os.path.join(cfg['explain_dir'], feat_shap_filename)

        with shap_cols[idx % len(shap_cols)]:
            st.markdown(f"#### Feature: `{feat_name}`")
            st.space("small")
            if os.path.exists(feat_shap_img):
                st.image(feat_shap_img, use_container_width=True, caption=f"SHAP Summary Beeswarm for {feat_name}")
            else:
                st.warning(
                    f'Feature SHAP summary plot for "{feat_name}" not found at searched location: `{to_rel_path(feat_shap_img)}` (`{feat_shap_img}`)')

    st.markdown("---")
    st.subheader("Unified All-Features Summary Beeswarm")
    st.markdown("""
        **What you can observe:**
        A comprehensive global ranking combining both graph node attributes (e.g., expression, copy number) and eventual tabular sample metadata/clinical covariates in a single unified view.
        """)
    st.space(size="small")

    all_shap_filename = get_all_shap_png_filename(cfg['prefix'])
    all_shap_img = os.path.join(cfg['explain_dir'], all_shap_filename)
    if os.path.exists(all_shap_img):
        st.image(all_shap_img, use_container_width=True)
    else:
        st.warning(
            f"Unified SHAP Summary image not found at searched location: `{to_rel_path(all_shap_img)}` (`{all_shap_img}`)")

    st.markdown("---")
    st.subheader("📋 Inspect Complete Tabular TSV Outputs")
    st.markdown("""
        **What you can observe:**
        Inspect and download raw TSV tables containing calculated SHAP scores and saliency values per gene for downstream statistical analysis.
        """)
    st.space(size="small")

    tsv_options = [
        get_all_shap_tsv_filename(cfg['prefix']),
        get_gene_attention_map_tsv_filename(cfg['prefix'])
    ]
    for feat_name in active_node_features:
        tsv_options.append(get_gene_shap_tsv_filename(cfg['prefix'], feat_name))

    tsv_file_choice = st.selectbox("Select TSV Data File to Inspect:", tsv_options)
    target_tsv_path = os.path.join(cfg['explain_dir'], tsv_file_choice)
    df_tsv = load_tsv(target_tsv_path)

    if df_tsv is not None:
        if user_genes and ('gene_name' in df_tsv.columns or 'gene_id' in df_tsv.columns):
            col_target = 'gene_name' if 'gene_name' in df_tsv.columns else 'gene_id'
            df_filtered = df_tsv[df_tsv[col_target].astype(str).str.upper().isin([g.upper() for g in user_genes])]
            st.dataframe(df_filtered, use_container_width=True)
            st.caption(f"Displaying {len(df_filtered)} of {len(df_tsv)} total records based on active gene filter.")
        else:
            st.dataframe(df_tsv, use_container_width=True)
        st.download_button(
            label="💾 Download Filtered Table (TSV)",
            data=df_tsv.to_csv(sep='\t', index=False),
            file_name=tsv_file_choice,
            mime='text/tab-separated-values'
        )
    else:
        st.warning(
            f'Selected TSV file "{tsv_file_choice}" not found at searched location: `{to_rel_path(target_tsv_path)}` (`{target_tsv_path}`)')

# =============================================================================
# TAB 3: INTERACTIVE PPI GRAPH & ATTENTION MAP
# =============================================================================
with tab_graph:
    st.header("Interactive PPI Subgraph Navigation (Attention Weights)")
    st.markdown("""
        This interactive network visualizes Graph Attention Network (GAT) weights learned over graph edges.

        **What you can observe:**
        - **Edge Thickness:** Proportional to learned attention intensity between virtually interacting genes (thicker edges indicate stronger GNN information flow).
        - **Node Sizes & Colors:** Represent gene saliency scores, node degree (connectivity), or structural clusters depending on the selected coloring mode.
        - **Interactivity:** Drag nodes to inspect subnetworks, adjust the edge threshold slider, or hover over nodes and edges to reveal exact attention values.
        """)

    edges_filename = get_edge_attention_tsv_filename(cfg['prefix'])
    edges_path = os.path.join(cfg['explain_dir'], edges_filename)
    saliency_filename = get_gene_attention_map_tsv_filename(cfg['prefix'])
    saliency_path = os.path.join(cfg['explain_dir'], saliency_filename)

    edges_df = load_tsv(edges_path)
    saliency_df = load_tsv(saliency_path)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        top_n_edges = st.slider("Number of top interaction edges to display:", min_value=10, max_value=300, value=100,
                                step=10)
    with col_ctrl2:
        color_strategy = st.selectbox("Node Coloring Strategy:",
                                      ["Saliency Score", "Node Degree (Number of Connections)",
                                       "Connected Component (Clusters)"])
    with col_ctrl3:
        enable_physics = st.checkbox("Enable Dynamic Physics", value=False)

    if edges_df is not None:
        user_genes_tuple = tuple(user_genes) if user_genes else None
        html_graph = build_optimized_pyvis_subgraph(
            df_edges=edges_df,
            target_genes_tuple=user_genes_tuple,
            top_n=top_n_edges,
            saliency_df=saliency_df,
            color_mode=color_strategy,
            enable_physics=enable_physics
        )
        if html_graph:
            components.html(html_graph, height=670, scrolling=False)
        else:
            st.error("Error generating PyVis interactive graph.")
    else:
        st.error(
            f"❌ Edge attention TSV file not found at searched location: `{to_rel_path(edges_path)}` (`{edges_path}`)")

# =============================================================================
# TAB 4: GENE FEATURE BOXPLOTS
# =============================================================================
with tab_boxplots:
    st.subheader("🔥 2D Gene x Feature Input Attention Heatmap")
    st.markdown("""
        **What you can observe:**
        This 2D heatmap displays multi-modal input attention across top-ranked genes and feature modalities (e.g., Gene Expression vs. Log2 Copy Number), revealing which specific feature type drives node saliency.
        """)
    heatmap_filename = get_heatmap_png_filename(cfg['prefix'])
    heatmap_img_path = os.path.join(cfg['explain_dir'], heatmap_filename)
    if os.path.exists(heatmap_img_path):
        st.image(heatmap_img_path, use_container_width=True)
    else:
        st.warning(
            f"Gene x Feature Attention Heatmap image not found at searched location: `{to_rel_path(heatmap_img_path)}` (`{heatmap_img_path}`)")

    st.markdown("---")
    st.subheader("📊 Gene Feature Distribution Boxplots")
    st.markdown("""
        **What you can observe:**
        Boxplots illustrate raw input feature distributions across outcome target labels for top-prioritized genes. 
        This allows verification of feature directionality and biological separation across target groups.
        """)

    for feat in active_node_features:
        boxplot_filename = get_top5_boxplot_png_filename(cfg['prefix'], feat)
        boxplot_default_img = os.path.join(cfg['explain_dir'], boxplot_filename)
        if os.path.exists(boxplot_default_img):
            st.image(boxplot_default_img, use_container_width=True)
            st.caption(f"Default top 5 gene {feat} boxplots loaded from output directory.")
        else:
            st.info(
                f"Top 5 genes feature boxplot image not found at searched location: `{to_rel_path(boxplot_default_img)}` (`{boxplot_default_img}`). Run post-hoc explainability script to generate it.")

# =============================================================================
# TAB 5: PATHWAY ENRICHMENT
# =============================================================================
with tab_enrich:
    st.header("Functional Pathway Enrichment Analysis (KEGG & GSEA)")

    st.markdown("""
    This section presents **functional pathway enrichment analysis** (Over-Representation Analysis via KEGG, Reactome, and GO, alongside Preranked GSEA) computed on the top high-saliency genes prioritized by the model.

    **What you can observe:**
    - **Enriched Biological Pathways:** Identifies key biological mechanisms, signaling cascades, and metabolic pathways (e.g., DNA damage repair, cell cycle control, or immune signaling) that are statistically over-represented among the model's most influential genes.
    - **Higher-Level Biological Context:** Translates individual gene-level attention scores and feature attributions into systemic biological modules, providing mechanistic insights into what drives the model's tissue classification or patient outcome predictions.
    """)
    st.markdown("---")

    kegg_img_path = os.path.join(cfg['explain_dir'], FN_KEGG_BARPLOT_PNG)
    if os.path.exists(kegg_img_path):
        st.image(kegg_img_path, use_container_width=True)
    else:
        st.warning(
            f"KEGG pathway barplot image not found at searched location: `{to_rel_path(kegg_img_path)}` (`{kegg_img_path}`)")

    kegg_tsv_path = os.path.join(cfg['explain_dir'], FN_KEGG_TOP_PATHWAYS_TSV)
    df_kegg = load_tsv(kegg_tsv_path)
    if df_kegg is not None:
        st.subheader("Top Enriched KEGG Pathways")

        old_cols = [c for c in df_kegg.columns if 'old' in c.lower()]
        df_kegg_display = df_kegg.drop(columns=old_cols)

        kegg_column_config = {
            "Gene_set": st.column_config.TextColumn(
                "Gene Set",
                help="Source biological pathway library database (e.g., KEGG_2021_Human)"
            ),
            "Term": st.column_config.TextColumn(
                "Pathway Term",
                help="The name of the biological KEGG pathway"
            ),
            "Overlap": st.column_config.TextColumn(
                "Overlap",
                help="Ratio between the number of high-saliency genes found and the total pathway size"
            ),
            "P-value": st.column_config.NumberColumn(
                "P-value",
                format="%.4e",
                help="Unadjusted p-value indicating statistical significance from Fisher's exact test"
            ),
            "Adjusted P-value": st.column_config.NumberColumn(
                "Adjusted P-value",
                format="%.4e",
                help="False Discovery Rate (FDR) / Benjamini-Hochberg adjusted p-value for multiple testing correction"
            ),
            "Odds Ratio": st.column_config.NumberColumn(
                "Odds Ratio",
                format="%.3f",
                help="Measure of over-representation strength compared to random expectation"
            ),
            "Combined Score": st.column_config.NumberColumn(
                "Combined Score",
                format="%.2f",
                help="Enrichr combined score computed as c = ln(p-value) * z-score"
            ),
            "Genes": st.column_config.TextColumn(
                "Genes",
                help="List of high-saliency target genes identified within this specific pathway"
            )
        }

        st.dataframe(
            df_kegg_display.head(15),
            column_config=kegg_column_config,
            use_container_width=True
        )
    else:
        st.info(
            f"KEGG pathways TSV table not found at searched location: `{to_rel_path(kegg_tsv_path)}` (`{kegg_tsv_path}`)")


# =============================================================================
# TAB 6 (MIL-ONLY): TISSUE ATTENTION DISTRIBUTION
# =============================================================================
if cfg['prefix'] == 'mil':
    with tab_tissue_attn:
        st.header("🧪 MIL Attention Weights Aggregation Across Tissue Types")

        st.markdown("""
        This section presents the **aggregation of MIL attention weights** ($\alpha_i$) assigned across tissue sample types 
        (e.g., `Fluid` / ascites vs. `Solid_Metastasis` vs. `Solid_Pelvic`).

        **Methodological Note on Single-Sample Bias:**
        In Gated Attention MIL, the softmax normalizes weights per patient ($\sum \alpha_i = 1.0$). 
        In single-sample bags ($N=1$), attention is artificially forced to $100\%$ ($1.0$). 
        To evaluate **true model preference**, filter by **Multi-Tissue Mixed Bags**, where multiple tissue types co-occur in the same patient.
        """)
        st.markdown("---")

        tissue_tsv_path = os.path.join(cfg['explain_dir'], FN_MIL_TISSUE_ATTENTION_AGGREGATION)
        df_tissue_att = load_tsv(tissue_tsv_path)

        if df_tissue_att is not None and not df_tissue_att.empty:
            # Radio button per selezionare la coorte di analisi
            cohort_filter = st.radio(
                "Select Patient Cohort Filter for Analysis:",
                ["Multi-Tissue Mixed Bags (Unbiased Competition, N > 1)", "All Patients (Includes Single-Sample Bags)"],
                horizontal=True
            )

            is_mixed = "Unbiased" in cohort_filter

            mean_col = 'mean_attention_mixed' if is_mixed and 'mean_attention_mixed' in df_tissue_att.columns else 'mean_attention_all'
            pct_col = 'pct_attention_mixed' if is_mixed and 'pct_attention_mixed' in df_tissue_att.columns else 'pct_attention_all'
            cnt_col = 'sample_count_mixed' if is_mixed and 'sample_count_mixed' in df_tissue_att.columns else 'sample_count_all'

            df_sorted = df_tissue_att.sort_values(by=mean_col, ascending=False)

            # Metric Cards
            col1, col2, col3 = st.columns(3)
            top_tissue = df_sorted.iloc[0]['tissue_type']
            top_val = df_sorted.iloc[0][mean_col]
            top_pct = df_sorted.iloc[0][pct_col]

            col1.metric("Top Attended Tissue", str(top_tissue))
            col2.metric("Mean Attention Weight", f"{top_val:.4f}")
            col3.metric("Relative Attention Share", f"{top_pct:.1f}%")

            st.markdown("---")
            st.subheader(f"📊 Attention Distribution ({'Mixed Bags Only' if is_mixed else 'All Patients'})")

            col_chart, col_pct = st.columns(2)
            with col_chart:
                st.markdown("##### Mean Attention Weight ($\mu$)")
                st.bar_chart(df_sorted.set_index('tissue_type')[mean_col], color="#4F81BD")

            with col_pct:
                st.markdown("##### Relative Share (%)")
                st.bar_chart(df_sorted.set_index('tissue_type')[pct_col], color="#C0504D")

            st.markdown("---")
            st.subheader("📋 Aggregation Summary Table")
            st.dataframe(df_tissue_att, use_container_width=True)

            # Ispezione dettaglio per paziente
            raw_patient_tsv = os.path.join(cfg['explain_dir'], "mil_patient_tissue_attention_raw.tsv")
            df_raw_patient = load_tsv(raw_patient_tsv)
            if df_raw_patient is not None:
                with st.expander("🔍 Inspect Patient-Level Raw Data & Filter Multi-Tissue Bags"):
                    st.caption(
                        "Showing raw attention weights per sample alongside bag size and number of unique tissues per patient.")
                    st.dataframe(df_raw_patient, use_container_width=True)
                    st.download_button(
                        label="💾 Download Patient-Level Raw Attention Weights (TSV)",
                        data=df_raw_patient.to_csv(sep='\t', index=False),
                        file_name="mil_patient_tissue_attention_raw.tsv",
                        mime='text/tab-separated-values'
                    )
        else:
            st.warning(
                f"❌ Tissue attention aggregation file not found at searched location: `{to_rel_path(tissue_tsv_path)}` (`{tissue_tsv_path}`). Please run the MIL explainability pipeline from the sidebar to generate it.")
