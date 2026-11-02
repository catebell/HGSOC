import os
import numpy as np
import pandas as pd
import torch

import config_file_tree

path_ppi_file = config_file_tree.STRING_PPI_FILE
path_protein_gene_file = config_file_tree.STRING_ALIASES_FILE
driver_genes_filepath = config_file_tree.DRIVER_GENES_FILE

path_top_genes_file = os.path.join(config_file_tree.PROCESSED_SAMPLES_DIR, 'selected_genes.tsv')

TOP_RNA_GENES = 2500
TOP_CN_GENES = 1000

# genes (IFN-I / JAK-STAT / Hypoxia / oncogenes) from Daria Afenteva paper https://doi.org/10.1016/j.xcrm.2025.102316
ifn_genes = ['STAT1', 'STAT2', 'IRF1', 'IRF2', 'IRF9', 'ISG15', 'CXCL10', 'CXCL11', 'CMPK2', 'IFIT1', 'IFIT3', 'OAS1', 'CDKN2A',
             'CDKN2B', 'IFNA1', 'IFNB1', 'JAK1', 'JAK2', 'MX1', 'IFIT2', 'OAS2', 'USP18', 'HLA-A', 'HLA-B', 'HLA-C', 'B2M', 'HIF1A', 'VEGFA', 'CA9', 'BRAF', 'CCNE1', 'MYC', 'KRAS', 'NF1', 'TP53', 'BRCA1', 'BRCA2', 'RAD51C', 'RAD51D']


# FILTER PARAMETERS FOR 1ST-DEGREE NEIGHBORS
MIN_RNA_VARIANCE = 0.0001        # Exclude neighbors with 0 or near-zero variance
MIN_CANDIDATE_CONNECTIONS = 2   # Require neighbor to connect to at least N candidate genes (Bridge Nodes)

# parameters for ppi edges
PPI_MIN_SCORE = 0.9

# parameters for hybrid ppi-copathway edges
HYBRID_MIN_PPI_SCORE = 0.70  # Minimum physical confidence in STRING (0.70 = high confidence)
HYBRID_MIN_KEGG_OVERLAP = 1  # At least 1 KEGG pathway shared


def _get_candidate_genes(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    """ Select top genes based on RNA expression and CNV variance plus driver genes. """

    common_genes = df_expr.index.intersection(primary_cn_pivot.index)
    print(f"Common Genes in expr and cn tables: {len(common_genes)}")

    df_expr_sub = df_expr.loc[common_genes, :]
    primary_cn_sub = primary_cn_pivot.loc[common_genes, :]

    print("Computing RNA expression variance...")
    rna_variances = df_expr_sub.var(axis=1)
    top_rna_series = rna_variances.nlargest(TOP_RNA_GENES)
    top_rna_genes_set = set(top_rna_series.index)
    print(f"Selected top {len(top_rna_genes_set)} genes by RNA expression variance.\n")

    print("Computing CN values variance...")
    cn_variances = primary_cn_sub.var(axis=1)
    top_cn_series = cn_variances.nlargest(TOP_CN_GENES)
    top_cn_genes_set = set(top_cn_series.index)
    print(f"Selected top {len(top_cn_genes_set)} genes by CN values variance.\n")

    selected_genes_union = top_rna_genes_set.union(top_cn_genes_set)
    overlap_count = len(top_rna_genes_set.intersection(top_cn_genes_set))

    print("\n--- GENE SELECTION SUMMARY ---")
    print(f"Top RNA genes count:        {len(top_rna_genes_set)}")
    print(f"Top CN genes count:         {len(top_cn_genes_set)}")
    print(f"Overlapping genes count:    {overlap_count}")
    print(f"Unique Selected genes:   {len(selected_genes_union)}\n")

    print(f"Adding Driver genes from '{driver_genes_filepath}'...")
    df_drivers = pd.read_csv(driver_genes_filepath, sep='\t')
    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}
    driver_genes_set = set(df_drivers['Symbol'].map(symbol_to_ensg).dropna())
    print(f"Found {len(driver_genes_set)} unique driver gene symbols in loaded file.")

    candidate_genes_set = top_rna_genes_set.union(top_cn_genes_set).union(driver_genes_set)
    print(f"Driver genes count:          {len(driver_genes_set)}")
    new_drivers_added = len(driver_genes_set - top_rna_genes_set - top_cn_genes_set)
    print(f"New Driver genes added:      {new_drivers_added}")
    print(f"Initial total candidate genes (RNA + CN + Drivers): {len(candidate_genes_set)}\n")

    print(f"Adding eventual missing genes for IFN-I response (from Daria Afenteva paper about chemio-refractory)...")
    print(f"Genes loaded: {ifn_genes}")
    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}
    ifn_genes_set = set(pd.Series(ifn_genes).map(symbol_to_ensg).dropna())
    print(f"Mapped {len(ifn_genes_set)}/{len(ifn_genes)} unique gene symbols.")

    genes_missing = len(ifn_genes_set - candidate_genes_set)
    print(f"New paper-specific genes added:      {genes_missing}")
    candidate_genes_set = candidate_genes_set.union(ifn_genes_set)
    print(f"Initial total candidate genes (RNA + CN + Drivers + paper-genes): {len(candidate_genes_set)}\n")

    return common_genes, rna_variances, cn_variances, top_rna_genes_set, top_cn_genes_set, driver_genes_set, candidate_genes_set


def _save_selected_genes_tsv(final_selected_genes, top_rna_genes_set, top_cn_genes_set, driver_genes_set, rna_variances, cn_variances, ensg_to_symbol_map):
    # Save output selected genes table
    rows = []
    for gene_id in sorted(final_selected_genes):
        in_rna = gene_id in top_rna_genes_set
        in_cn = gene_id in top_cn_genes_set
        in_driver = gene_id in driver_genes_set

        if in_rna and in_cn:
            reason = 'both'
        elif in_rna:
            reason = 'RNA_variance'
        elif in_cn:
            reason = 'CN_variance'
        elif in_driver:
            reason = 'driver_added'
        else:
            reason = '1st_degree_ppi_neighbor'

        gene_name = ensg_to_symbol_map.get(gene_id, 'N/A')

        rows.append({
            'gene_id': gene_id,
            'gene_name': gene_name,
            'rna_variance': rna_variances.get(gene_id, np.nan),
            'cn_variance': cn_variances.get(gene_id, np.nan),
            'selection_reason': reason
        })

    df_selected = pd.DataFrame(rows).sort_values(by='rna_variance', ascending=False)
    os.makedirs(config_file_tree.PROCESSED_SAMPLES_DIR, exist_ok=True)
    df_selected.to_csv(path_top_genes_file, sep='\t', index=False)
    print(f"Saved selected genes to: '{path_top_genes_file}' (Shape: {df_selected.shape})\n")


def _build_ppi_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    """ Expand candidate genes with 1st-degree PPI neighbors and save selected_genes.tsv, and build edge index. """

    print("\n[BUILDING GRAPH] TOPOLOGY: STRING PPI")
    common_genes, rna_variances, cn_variances, top_rna_genes_set, top_cn_genes_set, driver_genes_set, candidate_genes_set = _get_candidate_genes(
        df_expr, primary_cn_pivot, ensg_to_symbol_map
    )

    # adding neighbors

    print("Reading STRING PPI and Alias files for 1st-degree neighbor expansion...")
    df_protein_gene = pd.read_csv(path_protein_gene_file, sep='\t', usecols=['#string_protein_id', 'alias'], dtype={
        '#string_protein_id': 'category', 'alias': 'string'
    }).rename(columns={'#string_protein_id': 'protein_id', 'alias': 'gene_id'})

    df_ppi = pd.read_csv(path_ppi_file, sep=' ', dtype={
        'protein1': 'category', 'protein2': 'category', 'combined_score': 'int'
    })

    df_ppi['combined_score'] = df_ppi['combined_score'] / 1000  # remap in [0,1] interval
    df_ppi.drop(df_ppi[df_ppi['combined_score'] < PPI_MIN_SCORE].index, inplace=True)
    print(f"Kept only probability score > {PPI_MIN_SCORE}.")

    # Filter protein-gene mapping for genes present in common_genes
    df_protein_gene_filtered = df_protein_gene[df_protein_gene['gene_id'].isin(common_genes)].dropna().drop_duplicates(subset=['gene_id', 'protein_id']).reset_index(drop=True)
    # 'alias' contains also the ENSGs (gene_id)

    # Get protein IDs corresponding to candidate_genes_set
    candidate_proteins = set(df_protein_gene_filtered[df_protein_gene_filtered['gene_id'].isin(candidate_genes_set)]['protein_id'])

    # Retrieve PPI interactions where AT LEAST ONE protein is in candidate_proteins
    ppi_candidate_edges = df_ppi[
        df_ppi['protein1'].isin(candidate_proteins) | df_ppi['protein2'].isin(candidate_proteins)
    ].copy()

    # Map proteins to gene_ids
    prot_to_gene_map = dict(zip(df_protein_gene_filtered['protein_id'], df_protein_gene_filtered['gene_id']))
    ppi_candidate_edges['gene1'] = ppi_candidate_edges['protein1'].map(prot_to_gene_map)
    ppi_candidate_edges['gene2'] = ppi_candidate_edges['protein2'].map(prot_to_gene_map)

    # Drop unmapped genes and remove self-loops (GATConv handles them with add_self_loops=True)
    ppi_candidate_edges = ppi_candidate_edges.dropna(subset=['gene1', 'gene2'])
    ppi_candidate_edges = ppi_candidate_edges[ppi_candidate_edges['gene1'] != ppi_candidate_edges['gene2']]


    # Identify potential 1st-degree PPI neighbor genes
    all_neighbors = (set(ppi_candidate_edges['gene1']).union(set(ppi_candidate_edges['gene2']))) - candidate_genes_set
    print(f"Total 1st-degree PPI neighbors found: {len(all_neighbors)}")

    # CONDITION 1: Filter neighbors by RNA Expression Variance (> MIN_RNA_VARIANCE)
    valid_variance_neighbors = {g for g in all_neighbors if rna_variances.get(g, 0.0) > MIN_RNA_VARIANCE}
    print(f"Neighbors after RNA variance filtering (> {MIN_RNA_VARIANCE}): {len(valid_variance_neighbors)}")

    # CONDITION 2: Filter neighbors by degree of connectivity to candidate genes (Bridge Nodes)
    # Count how many UNIQUE candidate genes each neighbor is connected to
    candidate_edges_subset = ppi_candidate_edges[
        (ppi_candidate_edges['gene1'].isin(valid_variance_neighbors) & ppi_candidate_edges['gene2'].isin(
            candidate_genes_set)) |
        (ppi_candidate_edges['gene2'].isin(valid_variance_neighbors) & ppi_candidate_edges['gene1'].isin(
            candidate_genes_set))
        ]

    neighbor_conn_counts = {}
    for _, row in candidate_edges_subset.iterrows():
        g1, g2 = row['gene1'], row['gene2']
        neighbor = g1 if g1 in valid_variance_neighbors else g2
        candidate = g2 if g1 in valid_variance_neighbors else g1

        if neighbor not in neighbor_conn_counts:
            neighbor_conn_counts[neighbor] = set()
        neighbor_conn_counts[neighbor].add(candidate)

    filtered_neighbor_genes = {
        g for g, connected_candidates in neighbor_conn_counts.items()
        if len(connected_candidates) >= MIN_CANDIDATE_CONNECTIONS
    }

    print(
        f"Neighbors after bridge/connectivity filtering (>= {MIN_CANDIDATE_CONNECTIONS} candidate connections): {len(filtered_neighbor_genes)}")

    final_selected_genes = candidate_genes_set.union(filtered_neighbor_genes)
    print(f"Total Unique Selected Genes after conditional 1st-degree expansion: {len(final_selected_genes)}\n")

    _save_selected_genes_tsv(final_selected_genes, top_rna_genes_set, top_cn_genes_set, driver_genes_set, rna_variances,
                             cn_variances, ensg_to_symbol_map)

    # Build final edge index between selected genes
    final_ppi_edges = ppi_candidate_edges[
        ppi_candidate_edges['gene1'].isin(final_selected_genes) & ppi_candidate_edges['gene2'].isin(final_selected_genes)
    ].drop_duplicates(subset=['gene1', 'gene2'])

    gene_to_num_idx = {gene: idx for idx, gene in enumerate(sorted(final_selected_genes))}

    edge_src = final_ppi_edges['gene1'].map(gene_to_num_idx).values
    edge_dst = final_ppi_edges['gene2'].map(gene_to_num_idx).values

    e_index = torch.tensor(np.array([edge_src, edge_dst]), dtype=torch.long)

    print(f"Genes in final graph: {len(gene_to_num_idx)}")
    print(f"Directed PPI edges found: {len(final_ppi_edges)}\n")

    return e_index, gene_to_num_idx


def _build_omnipath_kegg_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    print("\n[BUILDING GRAPH] TOPOLOGY: OMNIPATH (KEGG / REACTOME)")
    try:
        import omnipath as op
    except ImportError:
        raise ImportError("'omnipath' library required if GRAPH_EDGES_TYPE='omnipath_kegg'.")

    common_genes, rna_variances, cn_variances, top_rna_genes_set, top_cn_genes_set, driver_genes_set, candidate_genes_set = _get_candidate_genes(
        df_expr, primary_cn_pivot, ensg_to_symbol_map
    )

    print("Fetching interactions from OmniPath (filtering for KEGG & Reactome)...")
    try:
        interactions = op.interactions.AllInteractions.get(genesymbols=True)
    except Exception:
        interactions = op.interactions.AllInteractions.get()

    kegg_interactions = interactions[interactions['sources'].str.contains('KEGG|Reactome', na=False)].copy()

    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}
    kegg_interactions['gene1'] = kegg_interactions['source_genesymbol'].map(symbol_to_ensg)
    kegg_interactions['gene2'] = kegg_interactions['target_genesymbol'].map(symbol_to_ensg)

    kegg_interactions = kegg_interactions.dropna(subset=['gene1', 'gene2'])
    kegg_interactions = kegg_interactions[
        kegg_interactions['gene1'].isin(common_genes) & kegg_interactions['gene2'].isin(common_genes)
    ]
    kegg_interactions = kegg_interactions[kegg_interactions['gene1'] != kegg_interactions['gene2']]

    # 1st-degree expansion tra i geni candidati
    candidate_edges = kegg_interactions[
        kegg_interactions['gene1'].isin(candidate_genes_set) | kegg_interactions['gene2'].isin(candidate_genes_set)
    ].copy()

    all_neighbors = (set(candidate_edges['gene1']).union(set(candidate_edges['gene2']))) - candidate_genes_set
    valid_variance_neighbors = {g for g in all_neighbors if rna_variances.get(g, 0.0) > MIN_RNA_VARIANCE}

    candidate_edges_subset = candidate_edges[
        (candidate_edges['gene1'].isin(valid_variance_neighbors) & candidate_edges['gene2'].isin(candidate_genes_set)) |
        (candidate_edges['gene2'].isin(valid_variance_neighbors) & candidate_edges['gene1'].isin(candidate_genes_set))
    ]

    neighbor_conn_counts = {}
    for _, row in candidate_edges_subset.iterrows():
        g1, g2 = row['gene1'], row['gene2']
        neighbor = g1 if g1 in valid_variance_neighbors else g2
        candidate = g2 if g1 in valid_variance_neighbors else g1

        if neighbor not in neighbor_conn_counts:
            neighbor_conn_counts[neighbor] = set()
        neighbor_conn_counts[neighbor].add(candidate)

    filtered_neighbor_genes = {
        g for g, connected_candidates in neighbor_conn_counts.items()
        if len(connected_candidates) >= MIN_CANDIDATE_CONNECTIONS
    }

    final_selected_genes = candidate_genes_set.union(filtered_neighbor_genes)
    _save_selected_genes_tsv(final_selected_genes, top_rna_genes_set, top_cn_genes_set, driver_genes_set, rna_variances, cn_variances, ensg_to_symbol_map)

    final_edges = candidate_edges[
        candidate_edges['gene1'].isin(final_selected_genes) & candidate_edges['gene2'].isin(final_selected_genes)
    ].drop_duplicates(subset=['gene1', 'gene2'])

    gene_to_num_idx = {gene: idx for idx, gene in enumerate(sorted(final_selected_genes))}

    edge_list = []
    for _, row in final_edges.iterrows():
        u = gene_to_num_idx[row['gene1']]
        v = gene_to_num_idx[row['gene2']]
        edge_list.append((u, v))
        # if interaction is not directed, made bidirectional
        if not row.get('is_directed', True):
            edge_list.append((v, u))

    edge_list = list(set(edge_list))
    e_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    print(f"Genes in final graph: {len(gene_to_num_idx)}")
    print(f"OmniPath/KEGG edges found: {e_index.size(1)}\n")

    return e_index, gene_to_num_idx


def _build_kegg_copathway_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    print("\n[BUILDING GRAPH] TOPOLOGY: KEGG CO-PATHWAY OVERLAP")
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("'gseapy' library required if GRAPH_EDGES_TYPE='kegg_copathway'.")

    common_genes, rna_variances, cn_variances, top_rna_genes_set, top_cn_genes_set, driver_genes_set, candidate_genes_set = _get_candidate_genes(
        df_expr, primary_cn_pivot, ensg_to_symbol_map
    )

    print("Fetching KEGG pathway gene sets via gseapy...")
    kegg_db = gp.get_library(name='KEGG_2021_Human', organism='Human')

    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}
    final_selected_genes = candidate_genes_set
    _save_selected_genes_tsv(final_selected_genes, top_rna_genes_set, top_cn_genes_set, driver_genes_set, rna_variances, cn_variances, ensg_to_symbol_map)

    gene_to_num_idx = {gene: idx for idx, gene in enumerate(sorted(final_selected_genes))}

    overlap_threshold = getattr(config_file_tree, 'KEGG_OVERLAP_THRESHOLD', 1)
    from collections import defaultdict
    pair_counts = defaultdict(int)

    for pathway_name, pathway_genes in kegg_db.items():
        ensg_genes = [symbol_to_ensg[g] for g in pathway_genes if g in symbol_to_ensg and symbol_to_ensg[g] in gene_to_num_idx]
        for i in range(len(ensg_genes)):
            for j in range(i + 1, len(ensg_genes)):
                u, v = gene_to_num_idx[ensg_genes[i]], gene_to_num_idx[ensg_genes[j]]
                pair_counts[(min(u, v), max(u, v))] += 1

    edge_list = []
    for (u, v), count in pair_counts.items():
        if count >= overlap_threshold:
            edge_list.append((u, v))
            edge_list.append((v, u))

    e_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

    print(f"Genes in final graph: {len(gene_to_num_idx)}")
    print(f"KEGG Co-pathway edges found (threshold >= {overlap_threshold}): {e_index.size(1)}\n")

    return e_index, gene_to_num_idx


def _build_hybrid_ppi_kegg_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    """
    Costruisce una topologia ibrida combinando STRING PPI (Interazioni Proteiche Fisiche)
    e KEGG Co-pathway (Co-appartenenza Funzionale).
    Un arco viene incluso solo se i due geni interagiscono fisicamente E condividono almeno un pathway KEGG.
    """
    print("\n[BUILDING GRAPH] TOPOLOGY: HYBRID (STRING PPI ∩ KEGG CO-PATHWAY)")
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("'gseapy' library required for hybrid topology.")

    # 1. Selezione geni candidati ed espansione con vicini di 1° grado via PPI
    common_genes, rna_variances, cn_variances, top_rna_genes_set, top_cn_genes_set, driver_genes_set, candidate_genes_set = _get_candidate_genes(
        df_expr, primary_cn_pivot, ensg_to_symbol_map
    )

    print("Caricamento interazioni da STRING PPI...")
    df_protein_gene = pd.read_csv(path_protein_gene_file, sep='\t', usecols=['#string_protein_id', 'alias'], dtype={
        '#string_protein_id': 'category', 'alias': 'string'
    }).rename(columns={'#string_protein_id': 'protein_id', 'alias': 'gene_id'})

    df_ppi = pd.read_csv(path_ppi_file, sep=' ', dtype={
        'protein1': 'category', 'protein2': 'category', 'combined_score': 'int'
    })
    df_ppi['combined_score'] = df_ppi['combined_score'] / 1000.0

    df_ppi = df_ppi[df_ppi['combined_score'] >= HYBRID_MIN_PPI_SCORE].copy()
    print(f"Mantenute interazioni PPI con combined_score >= {HYBRID_MIN_PPI_SCORE}")

    df_protein_gene_filtered = df_protein_gene[df_protein_gene['gene_id'].isin(common_genes)].dropna().drop_duplicates(
        subset=['gene_id', 'protein_id']).reset_index(drop=True)
    candidate_proteins = set(
        df_protein_gene_filtered[df_protein_gene_filtered['gene_id'].isin(candidate_genes_set)]['protein_id'])

    ppi_candidate_edges = df_ppi[
        df_ppi['protein1'].isin(candidate_proteins) | df_ppi['protein2'].isin(candidate_proteins)
        ].copy()

    prot_to_gene_map = dict(zip(df_protein_gene_filtered['protein_id'], df_protein_gene_filtered['gene_id']))
    ppi_candidate_edges['gene1'] = ppi_candidate_edges['protein1'].map(prot_to_gene_map)
    ppi_candidate_edges['gene2'] = ppi_candidate_edges['protein2'].map(prot_to_gene_map)
    ppi_candidate_edges = ppi_candidate_edges.dropna(subset=['gene1', 'gene2'])
    ppi_candidate_edges = ppi_candidate_edges[ppi_candidate_edges['gene1'] != ppi_candidate_edges['gene2']]

    # 2. Espansione condizionale dei vicini di 1° grado (Bridge Nodes)
    all_neighbors = (set(ppi_candidate_edges['gene1']).union(set(ppi_candidate_edges['gene2']))) - candidate_genes_set
    valid_variance_neighbors = {g for g in all_neighbors if rna_variances.get(g, 0.0) > MIN_RNA_VARIANCE}

    candidate_edges_subset = ppi_candidate_edges[
        (ppi_candidate_edges['gene1'].isin(valid_variance_neighbors) & ppi_candidate_edges['gene2'].isin(
            candidate_genes_set)) |
        (ppi_candidate_edges['gene2'].isin(valid_variance_neighbors) & ppi_candidate_edges['gene1'].isin(
            candidate_genes_set))
        ]

    neighbor_conn_counts = {}
    for _, row in candidate_edges_subset.iterrows():
        g1, g2 = row['gene1'], row['gene2']
        neighbor = g1 if g1 in valid_variance_neighbors else g2
        candidate = g2 if g1 in valid_variance_neighbors else g1

        if neighbor not in neighbor_conn_counts:
            neighbor_conn_counts[neighbor] = set()
        neighbor_conn_counts[neighbor].add(candidate)

    filtered_neighbor_genes = {
        g for g, connected_candidates in neighbor_conn_counts.items()
        if len(connected_candidates) >= MIN_CANDIDATE_CONNECTIONS
    }

    final_selected_genes = candidate_genes_set.union(filtered_neighbor_genes)
    _save_selected_genes_tsv(final_selected_genes, top_rna_genes_set, top_cn_genes_set, driver_genes_set, rna_variances,
                             cn_variances, ensg_to_symbol_map)

    # 3. Mappatura Geni KEGG Pathway
    print("Estrazione pathway KEGG da GSEAPy...")
    kegg_db = gp.get_library(name='KEGG_2021_Human', organism='Human')
    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}

    # Costruzione set di coppie di geni che condividono almeno un pathway KEGG
    from collections import defaultdict
    shared_kegg_pathways = defaultdict(int)

    for pathway_name, pathway_genes in kegg_db.items():
        ensg_genes = [symbol_to_ensg[g] for g in pathway_genes if
                      g in symbol_to_ensg and symbol_to_ensg[g] in final_selected_genes]
        for i in range(len(ensg_genes)):
            for j in range(i + 1, len(ensg_genes)):
                g1, g2 = ensg_genes[i], ensg_genes[j]
                pair = (min(g1, g2), max(g1, g2))
                shared_kegg_pathways[pair] += 1

    # 4. Intersezione tra Edges PPI e Co-appartenenza KEGG
    final_ppi_edges = ppi_candidate_edges[
        ppi_candidate_edges['gene1'].isin(final_selected_genes) & ppi_candidate_edges['gene2'].isin(
            final_selected_genes)
        ].drop_duplicates(subset=['gene1', 'gene2'])

    gene_to_num_idx = {gene: idx for idx, gene in enumerate(sorted(final_selected_genes))}

    hybrid_edge_list = []
    for _, row in final_ppi_edges.iterrows():
        g1, g2 = row['gene1'], row['gene2']
        pair = (min(g1, g2), max(g1, g2))

        # Filtro d'intersezione ibrido: Deve essere sia un PPI sia condividere pathway KEGG
        if shared_kegg_pathways.get(pair, 0) >= HYBRID_MIN_KEGG_OVERLAP:
            u, v = gene_to_num_idx[g1], gene_to_num_idx[g2]
            hybrid_edge_list.append((u, v))
            hybrid_edge_list.append((v, u))  # Rende il grafo non orientato

    hybrid_edge_list = list(set(hybrid_edge_list))

    if hybrid_edge_list:
        e_index = torch.tensor(hybrid_edge_list, dtype=torch.long).t().contiguous()
    else:
        e_index = torch.empty((2, 0), dtype=torch.long)

    print(f"Geni nel grafo finale: {len(gene_to_num_idx)}")
    print(f"Archi Ibridi (PPI ∩ KEGG) trovati: {e_index.size(1)}\n")

    return e_index, gene_to_num_idx


def build_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    """
    Dispatcher function that builds edge_index for PyTorch Geometric getting topology from config_file_tree.GRAPH_EDGES_TYPE ('ppi', 'omnipath_kegg', 'kegg_copathway').
    """
    edge_type = str(getattr(config_file_tree, 'GRAPH_EDGES_TYPE', 'ppi')).lower().strip()

    if edge_type in ['ppi', 'string']:
        return _build_ppi_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map)
    elif edge_type in ['omnipath_kegg', 'omnipath']:
        return _build_omnipath_kegg_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map)
    elif edge_type in ['kegg_copathway', 'kegg_co_pathway', 'kegg']:
        return _build_kegg_copathway_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map)
    elif edge_type in ['hybrid', 'hybrid_ppi_kegg', 'ppi_kegg']:
        return _build_hybrid_ppi_kegg_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map)
    else:
        print(f"[WARNING] GRAPH_EDGES_TYPE '{edge_type}' unrecognized. Using default 'ppi'.")
        return _build_ppi_edge_index(df_expr, primary_cn_pivot, ensg_to_symbol_map)


if __name__ == "__main__":
    #edge_index, gene_to_idx = build_edge_index()
    # print to verify that links go both ways (755 can be changed to any gene num_idx):
    # pd.DataFrame(edge_index.detach().numpy()).T[(pd.DataFrame(edge_index.detach().numpy()).T[0] == 755) | (pd.DataFrame(edge_index.detach().numpy()).T[1] == 755)]
    print()
