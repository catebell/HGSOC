from os import path, makedirs
from shutil import rmtree

import numpy as np
import torch
from numpy import column_stack
from torch import tensor, save
from torch_geometric.data import Data
from pandas import read_csv, DataFrame
from tqdm import tqdm

import PCA
import config_file_tree
from ppi_build import build_edge_index

""""" Data conversion in PyTorch Geometric Data objects that will be passed to the GNN.
Each graph (Data) will represent a single tissue (its genes = nodes, ppi = edges, expression values = node features).
Node features vector x layout:
- Feature 0: Gene Expression
- Feature 1: log2CopyNumber
- Feature 2: nMajor
- Feature 3: nMinor
- Feature 4: Loh
- Feature 5: breaksInGene
"""""


""""" paths and parameters """""

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
preprocessed_expr_matrix_filepath = path.join(preprocessed_dir, 'preprocessed_bulk_expr.tsv')
preprocessed_cn_filepath = path.join(preprocessed_dir, 'preprocessed_cn.tsv')
preprocessed_metadata_filepath = path.join(preprocessed_dir, 'preprocessed_samples_metadata.tsv')
preprocessed_clinical_filepath = path.join(preprocessed_dir, 'preprocessed_clinical_data.tsv')

driver_genes_filepath = config_file_tree.DRIVER_GENES_FILE

output_dir = config_file_tree.PROCESSED_SAMPLES_DIR
output_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR

output_selected_genes_filepath = path.join(output_dir, "selected_genes.tsv")
output_metadata_filepath = path.join(output_dir, "associated_metadata.tsv")

TOP_RNA_GENES = 5000
TOP_CN_GENES = 2000

CNV_FEATURE_COLS = ['log2CopyNumber', 'nMajor', 'nMinor', 'Loh', 'breaksInGene']


def filter_patients(df_clinical):
    """ Exclude some patients based on their features values. """

    # filter out patient based on 'Histology'
    if 'Histology' in df_clinical.columns:
        n_patients = len(df_clinical)
        hgsc_patients = df_clinical[df_clinical['Histology'] == 'high grade serous']['patient_id']
        df_clinical = df_clinical[df_clinical['patient_id'].isin(hgsc_patients)]
        print(
            f"\nDiscarded {n_patients - len(df_clinical)} patients with 'Histology' != 'high grade serous'.\n")
    return df_clinical


def filter_tissues(df_metadata, clinical_patients):
    """ Exclude some samples based on their features values. """

    n_before = len(df_metadata)
    df_metadata = df_metadata[df_metadata['patient_id'].isin(clinical_patients)]
    print(f"Discarded {n_before - len(df_metadata)} samples with patient_id not appearing in clinical file.")

    # filter based on phase label
    if 'phase_label' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['phase_label'] == 'Primary']
        print(f"Discarded {n_before - len(df_metadata)} samples with phase_label != 'Primary' (Interval, Relapse).")

    # filter based on class of sample
    if 'sample_class' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['sample_class'] == 'BulkTissue']
        print(f"Discarded {n_before - len(df_metadata)} samples with sample_class != 'BulkTissue' (CellLines, OrganoidLines, ...).")

    # filter based on treatment
    if 'treatment' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['treatment'] == 'Untreated']
        print(f"Discarded {n_before - len(df_metadata)} samples with treatment != 'Untreated'.")

    return df_metadata


def clean_cnv_outliers(cn_pivots, df_expr, df_metadata, primary_feat='log2CopyNumber'):
    """ QC check on specific feature to remove CNV outlier samples and clip extreme values. """
    primary_pivot = cn_pivots[primary_feat]
    sample_means = cn_pivots[primary_feat].mean(axis=0)
    floor_counts = (primary_pivot <= -5.0).sum(axis=0)

    bad_samples = sample_means[(sample_means < -0.20) | (floor_counts > 100)].index.tolist()

    if len(bad_samples) > 0:
        print(f"\n[QC] Found and discarded {len(bad_samples)} CNV outliers: {bad_samples}")
        for feature in cn_pivots:
            cn_pivots[feature] = cn_pivots[feature].drop(columns=bad_samples, errors='ignore')

        # Keep RNA and metadata perfectly aligned
        remaining_samples = cn_pivots[primary_feat].columns
        df_expr = df_expr[[c for c in df_expr.columns if c in remaining_samples]].copy()
        df_metadata = df_metadata.loc[df_metadata.index.intersection(remaining_samples)].copy()

    # Clip values for GNN stability
    if primary_feat in cn_pivots:
        cn_pivots[primary_feat] = cn_pivots[primary_feat].clip(lower=-3.0, upper=3.0)

    return cn_pivots, df_expr, df_metadata


def select_genes(df_expr, primary_cn_pivot, ensg_to_symbol_map):
    """ Select top genes based on RNA expression and CNV variance plus driver genes. """
    common_genes = df_expr.index.intersection(primary_cn_pivot.index)
    print(f"Common Genes in expr and cn tables: {len(common_genes)}")

    df_expr = df_expr.loc[common_genes, :]
    primary_cn_pivot = primary_cn_pivot.loc[common_genes, :]

    print(f"Computing RNA expression variance...")
    rna_variances = df_expr.var(axis=1)
    top_rna_series = rna_variances.nlargest(TOP_RNA_GENES)
    top_rna_genes_set = set(top_rna_series.index)
    print(f"Selected top {len(top_rna_genes_set)} genes by RNA expression variance.\n")

    print(f"Computing CN values variance...")
    #cn_variances = df_cn.groupby('gene_id', observed=True)['purifiedLogR'].var().dropna()
    cn_variances = primary_cn_pivot.var(axis=1)
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
    df_drivers = read_csv(driver_genes_filepath, sep='\t')
    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}
    driver_genes_set = set(df_drivers['Symbol'].map(symbol_to_ensg).dropna())

    print(f"Found {len(driver_genes_set)} unique driver gene symbols in file loaded.")
    selected_genes_union = top_rna_genes_set.union(top_cn_genes_set).union(driver_genes_set)
    print(f"Driver genes count:          {len(driver_genes_set)}")
    new_drivers_added = len(driver_genes_set - top_rna_genes_set - top_cn_genes_set)
    print(f"New Driver genes added:      {new_drivers_added}")
    print(f"Total Unique Selected genes:  {len(selected_genes_union)}\n")

    # output dataframe
    rows = []
    for gene_id in sorted(selected_genes_union):
        in_rna = gene_id in top_rna_genes_set
        in_cn = gene_id in top_cn_genes_set

        if in_rna and in_cn:
            reason = 'both'
        elif in_rna:
            reason = 'RNA_variance'
        elif in_cn:
            reason = 'CN_variance'
        else:
            reason = 'driver added'

        gene_name = ensg_to_symbol_map.get(gene_id, 'N/A')

        rows.append({
            'gene_id': gene_id,
            'gene_name': gene_name,
            'rna_variance': rna_variances.get(gene_id, np.nan),
            'cn_variance': cn_variances.get(gene_id, np.nan),
            'selection_reason': reason
        })

    df_selected = DataFrame(rows).sort_values(by='rna_variance', ascending=False)
    df_selected.to_csv(output_selected_genes_filepath, sep='\t', index=False)
    print(f"Saved selected genes to: '{output_selected_genes_filepath}' (Shape: {df_selected.shape})\n")


def generate_all_graphs(df_expr, cn_pivots, df_metadata):
    """ Build PyG graph objects storing multi-feature node tensors x. """
    print("Extracting edge index...")
    edge_index, gene_to_idx = build_edge_index()

    genes_ordered = sorted(gene_to_idx, key=gene_to_idx.get)  # sort to 0,1,2,...
    print(f"Number of nodes (genes) in graph: {len(genes_ordered)}\n")

    # saving gene node and symbol mapping file
    df_selected = read_csv(output_selected_genes_filepath, sep='\t')
    df_selected['node_index'] = df_selected['gene_id'].map(gene_to_idx)

    df_gene_mapping = DataFrame({
        'node_index': df_selected['node_index'].astype(int),
        'gene_id': df_selected['gene_id'],
        'gene_symbol': df_selected['gene_name']
    }).sort_values(by='node_index').reset_index(drop=True)

    gene_mapping_filepath = path.join(output_dir, "gene_node_mapping.tsv")
    df_gene_mapping.to_csv(gene_mapping_filepath, sep='\t', index=False)
    print(f"Saved gene node mapping reference to: '{gene_mapping_filepath}' (Shape: {df_gene_mapping.shape})\n")


    """"" aligning genes and samples between tables """""
    primary_cn = list(cn_pivots.values())[0]  # reference feature for cn
    matched_samples = [s for s in df_expr.columns if s in primary_cn.columns]
    print(f"\n[ALIGNMENT] Valid samples matched (RNA <-> CNV): {len(matched_samples)}")

    df_expr_aligned = df_expr[matched_samples].reindex(genes_ordered).fillna(0.0)  # missing nodes/genes filled with 0 expression
    print(f"    Aligned matrix shape: {df_expr_aligned.shape}\n")

    # Reindex each CNV pivot table to gene nodes and fill default values
    cn_aligned_dict = {}
    default_values = {
        'log2CopyNumber': 0.0,
        'nMajor': 1.0,
        'nMinor': 1.0,
        'Loh': 0.0,
        'breaksInGene': 0.0
    }

    for feature, pivot_df in cn_pivots.items():
        def_val = default_values.get(feature, 0.0)
        cn_aligned_dict[feature] = pivot_df[matched_samples].reindex(index=genes_ordered).fillna(def_val)
        print(f"    Aligned CN matrix of feature {feature} shape: {cn_aligned_dict[feature].shape}\n")

    feature_names = ['gene_expression'] + list(cn_aligned_dict.keys())
    num_features = len(feature_names)


    print("Generating PyTorch Geometric graph objects...")
    for sample_id in tqdm(matched_samples):
        # Build array list for column_stack
        feature_arrays = [df_expr_aligned[sample_id].values]  # expr values

        for feature in cn_aligned_dict:  # cn related values
            feature_arrays.append(cn_aligned_dict[feature][sample_id].values)

        # concatenating feature vectors for single sample -> shape: [Num_Genes, 2]
        node_features = column_stack(feature_arrays)
        x = tensor(node_features, dtype=torch.float32)

        graph = Data(
            x=x,
            edge_index=edge_index,
            sample_id=sample_id
        )

        save_path = path.join(output_graphs_dir, f"{sample_id}.pt")
        save(graph, save_path)

    print(f"Save completed for {len(matched_samples)} graphs in {output_graphs_dir}\n")
    print(f"Node feature tensor shape (x): [{len(genes_ordered)}, {num_features}]\n")
    print(f"Features mapped to node tensor: {feature_names}\n")

    df_metadata = df_metadata.loc[matched_samples,:]
    df_metadata.to_csv(output_metadata_filepath, sep='\t', index=True)
    print(f"Associated metadata saved to: {output_metadata_filepath}")

    return df_expr_aligned, cn_aligned_dict['log2CopyNumber']


if __name__ == "__main__":
    if path.exists(output_dir):
        rmtree(output_dir)
    makedirs(output_graphs_dir, exist_ok=True)

    metadata_table = read_csv(preprocessed_metadata_filepath, sep='\t', index_col=0)
    clinical_table = read_csv(preprocessed_clinical_filepath, sep='\t')

    clinical_table_filtered = filter_patients(clinical_table)
    valid_patients = clinical_table_filtered['patient_id']
    metadata_table_filtered = filter_tissues(metadata_table, valid_patients)
    valid_sample_ids = metadata_table_filtered.index.tolist()

    print("\nLoading preprocessed RNA expression matrix...")
    expr_table = read_csv(preprocessed_expr_matrix_filepath, sep="\t", index_col=0)
    print(f"    Original expression matrix shape: {expr_table.shape} (Genes x Samples)")
    expr_table_filtered = expr_table[[c for c in expr_table.columns if c in valid_sample_ids]].copy()
    print(f"    Filtered expression matrix shape: {expr_table_filtered.shape} (Genes x Samples)")

    print("\nLoading preprocessed CN values matrix...")
    df_cn = read_csv(preprocessed_cn_filepath, sep='\t', dtype={
        'gene_id': 'category',
        'symbol': 'category',
        'sample_id': 'category',
    })

    print("Pivoting cn table for selected features...")
    cn_pivots = {}
    for feat in CNV_FEATURE_COLS:
        if feat in df_cn.columns:
            pivot = df_cn.pivot_table(
                index='gene_id',
                columns='sample_id',
                values=feat,
                aggfunc='mean',
                observed=True,
            )
            cn_pivots[feat] = pivot[[c for c in pivot.columns if c in valid_sample_ids]].copy()
            print(f"    Pivot shape for '{feat}': {cn_pivots[feat].shape}")

    cn_pivots, expr_table_filtered, metadata_table_filtered = clean_cnv_outliers(
        cn_pivots, expr_table_filtered, metadata_table_filtered, primary_feat='log2CopyNumber'
    )

    ensg_to_symbol = df_cn.groupby('gene_id', observed=True)['symbol'].first().to_dict()
    select_genes(expr_table_filtered, cn_pivots['log2CopyNumber'], ensg_to_symbol)
    used_expr_df, used_cn_df = generate_all_graphs(expr_table_filtered, cn_pivots, metadata_table_filtered)

    if config_file_tree.RUN_PCA:
        out_dir_pca = path.join('processed_graphs', "PCA_data_used")
        makedirs(out_dir_pca, exist_ok=True)

        for feature_type, df in [('Expr', used_expr_df), ('CNV', used_cn_df)]:
            for group_col in ['cohort', 'tissue', 'tissue_type', 'phase_label', 'is_fluid']:
                if group_col in metadata_table_filtered.columns:
                    PCA.run_pca_analysis(df, output_metadata_filepath, out_dir_pca, feature_type, group_col)