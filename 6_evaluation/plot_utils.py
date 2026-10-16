import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
from matplotlib.colors import to_hex, to_rgb
from pyvis.network import Network

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


def plot_interactive_gene_network(tsv_path, output_html, top_n=30, saliency_tsv_path=None):
    """
    Crea un grafo interattivo HTML a schermo intero su sfondo bianco:
    - Archi spessi, marcati e proporzionali all'attention weight (4px - 12px).
    - Nodi sfumati in colori pastello tenui in base al punteggio di saliency.
    - Bordi e testo in grigio scuro (#444444 / #333333).
    """
    df = pd.read_csv(tsv_path, sep='\t').head(top_n)

    # 1. Carica i punteggi di saliency dei geni se il file esiste
    saliency_dict = {}
    if saliency_tsv_path and os.path.exists(saliency_tsv_path):
        df_sal = pd.read_csv(saliency_tsv_path, sep='\t')
        saliency_dict = dict(zip(df_sal['gene_name'], df_sal['saliency_score']))

    G = nx.Graph()

    # 2. Normalizzazione dello spessore degli archi (da 4.0px a 12.0px per marcata visibilità)
    min_w = df['attention_weight'].min()
    max_w = df['attention_weight'].max()
    denom_w = (max_w - min_w) if max_w != min_w else 1.0

    for _, row in df.iterrows():
        u = str(row['gene_symbol_a'])
        v = str(row['gene_symbol_b'])
        raw_w = float(row['attention_weight'])

        # Spessore marcato degli archi
        norm_width = 4.0 + 4.0 * ((raw_w - min_w) / denom_w)

        G.add_edge(
            u, v,
            weight=raw_w,
            width=norm_width,
            value=norm_width,  # Passaggio esplicito sia di width che di value per PyVis
            title=f"{u} - {v}\nAttention Weight: {raw_w:.5f}",
            color={'color': 'rgba(74, 123, 176, 0.85)', 'highlight': '#e63946'}  # Blu/grigio nitido ed evidente
        )

    # 3. Calcolo e normalizzazione Saliency per sfumatura colori pastello dei nodi
    node_saliencies = [saliency_dict.get(node, 0.0) for node in G.nodes()]
    min_sal = min(node_saliencies) if node_saliencies else 0.0
    max_sal = max(node_saliencies) if node_saliencies else 1.0
    sal_denom = (max_sal - min_sal) if max_sal != min_sal else 1.0

    # Gradient pastello tenue: Azzurro Pastello (#B3D9FF) -> Rosa Pastello (#FFB7B2)
    c_start = np.array(to_rgb("#B3D9FF"))
    c_end = np.array(to_rgb("#FFB7B2"))

    degrees = dict(G.degree())

    for node in G.nodes():
        deg = degrees[node]
        raw_sal = saliency_dict.get(node, min_sal)

        # Normalizzazione del valore di saliency sui nodi effettivi del grafo (range 0.0 - 1.0)
        norm_score = (raw_sal - min_sal) / sal_denom

        # Calcolo colore tenue interpolato
        pastel_rgb = c_start * (1.0 - norm_score) + c_end * norm_score
        color_hex = to_hex(pastel_rgb)

        # Dimensione nodo (tra 25px e 45px)
        node_size = 25 + 20 * norm_score

        G.nodes[node]['size'] = node_size
        G.nodes[node]['color'] = {
            'background': color_hex,
            'border': '#444444',  # Bordo grigio scuro
            'highlight': {'background': '#a8dda8', 'border': '#222222'}
        }
        G.nodes[node]['font'] = {
            'color': '#333333',  # Testo grigio scuro
            'size': 14,
            'face': 'arial',
            'strokeWidth': 2,  # Contorno bianco attorno alle scritte per leggibilità
            'strokeColor': '#ffffff'
        }
        G.nodes[node]['title'] = f"Gene: {node}\nConnessioni: {deg}\nSaliency: {raw_sal:.5f}"

    # 4. Inizializzazione PyVis su sfondo BIANCO (#ffffff)
    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333", directed=False)
    net.from_nx(G)

    # Configurazione della fisica di distribuzione dei nodi
    net.barnes_hut(gravity=-3000, central_gravity=0.2, spring_length=140, spring_strength=0.04)

    # Iniezione CSS per sfondo bianco totale a tutto schermo
    html_content = net.generate_html()
    custom_css = """
    <style>
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
            background-color: #ffffff !important;
        }
        #mynetwork {
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            background-color: #ffffff !important;
        }
    </style>
    </head>
    """
    html_content = html_content.replace("</head>", custom_css)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)


def plot_static_gene_network(tsv_path, output_png, top_n=30, saliency_tsv_path=None):
    """
    Genera un'immagine PNG statica a 300 DPI delle top N interazioni gene-gene:
    - Layout a spirale aurea (Phyllotaxis) che riempie il centro ed evita sovrapposizioni.
    - Sfumatura pastello tenue sui nodi basata sul punteggio di Saliency.
    - Archi ben spessi, bordi nodi grigio scuro (#444444) e testo grigio scuro (#222222).
    """
    import matplotlib.colors as mcolors

    df = pd.read_csv(tsv_path, sep='\t').head(top_n)

    # 1. Carica i punteggi di saliency dei geni se il file esiste
    saliency_dict = {}
    if saliency_tsv_path and os.path.exists(saliency_tsv_path):
        df_sal = pd.read_csv(saliency_tsv_path, sep='\t')
        saliency_dict = dict(zip(df_sal['gene_name'], df_sal['saliency_score']))

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            str(row['gene_symbol_a']),
            str(row['gene_symbol_b']),
            weight=float(row['attention_weight'])
        )

    # --- 2. LAYOUT A SPIRALE PER RIEMPIMENTO 2D UNIFORME ---
    components = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    pos = {}
    golden_angle = 137.50776405003785 * (np.pi / 180)  # Angolo aureo
    scale = 1.55  # Spaziatura ottima tra componenti

    for i, comp in enumerate(components):
        # Calcolo del centro di ciascun gruppo lungo la spirale di Fermat
        r = scale * np.sqrt(i + 0.5)
        theta = i * golden_angle
        center = np.array([r * np.cos(theta), r * np.sin(theta)])

        if len(comp) == 1:
            comp_pos = {list(comp.nodes())[0]: np.array([0.0, 0.0])}
        elif len(comp) == 2:
            n1, n2 = list(comp.nodes())
            angle_tangent = theta + np.pi / 2
            tangent = np.array([np.cos(angle_tangent), np.sin(angle_tangent)])
            comp_pos = {n1: -0.38 * tangent, n2: 0.38 * tangent}
        else:
            comp_pos = nx.spring_layout(comp, k=0.9, seed=42)

        for node, p in comp_pos.items():
            pos[node] = center + p

    # --- 3. COLORI PASTELLO E SALIENCY ---
    nodes = list(G.nodes())
    node_saliencies = [saliency_dict.get(node, 0.0) for node in nodes]
    min_sal = min(node_saliencies) if node_saliencies else 0.0
    max_sal = max(node_saliencies) if node_saliencies else 1.0
    sal_denom = (max_sal - min_sal) if max_sal != min_sal else 1.0

    # Gradient pastello tenue: Azzurro Pastello (#B3D9FF) -> Rosa Pastello (#FFB7B2)
    c_start = np.array(mcolors.to_rgb("#B3D9FF"))
    c_end = np.array(mcolors.to_rgb("#FFB7B2"))

    scores = [(saliency_dict.get(node, min_sal) - min_sal) / sal_denom for node in nodes]
    node_colors = [mcolors.to_hex(c_start * (1 - s) + c_end * s) for s in scores]

    # --- 4. SPESSORI DEGLI ARCHI E RENDERING GRAFICO ---
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w, max_w = min(weights), max(weights)
    denom_w = (max_w - min_w) if max_w != min_w else 1.0
    edge_widths = [2.5 + 3.0 * ((w - min_w) / denom_w) for w in weights]

    plt.figure(figsize=(12, 12), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # Archi marcati e ben visibili
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='#888888',
        alpha=0.75,
        ax=ax
    )

    # Nodi pastello con bordo grigio scuro
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=750,
        edgecolors='#444444',
        linewidths=1.2,
        alpha=0.95,
        ax=ax
    )

    # Etichette dei geni nitide in grigio scuro
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_family='sans-serif',
        font_weight='bold',
        font_color='#222222',
        ax=ax
    )

    plt.title(f"Top {top_n} Gene-Gene PPI Interactions", fontsize=15, fontweight='bold', pad=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()