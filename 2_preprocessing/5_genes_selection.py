import logging
from os import path
import pandas as pd
import numpy as np

import config_file_tree

import time
start_time = time.time()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


""""" paths and parameters """""

output_dir = config_file_tree.PREPROCESSED_DATA_DIR

rna_bulk_filepath = path.join(output_dir, "preprocessed_bulk_expr.tsv")
cn_filepath = path.join(output_dir, "preprocessed_cn.tsv")
driver_genes_filepath = config_file_tree.DRIVER_GENES_FILE

output_selected_genes_filepath = path.join(output_dir, "selected_genes.tsv")

TOP_RNA_GENES = 5000
TOP_CN_GENES = 2000


def select_genes():
    logging.info(f"Loading RNA expression matrix from '{rna_bulk_filepath}'...")
    df_rna = pd.read_csv(rna_bulk_filepath, sep='\t', index_col=0)
    print(f"Genes found in expression matrix: {len(df_rna)}")

    logging.info(f"Loading Copy Number data from '{cn_filepath}'...")
    df_cn = pd.read_csv(cn_filepath, sep='\t', dtype={
        'gene_id': 'category',
        'symbol': 'category',
        'sample': 'category',
        'purifiedLogR': 'float32'
    })
    df_cnv_pivot = df_cn.pivot_table(
        index='gene_id',
        columns='sample',
        values='purifiedLogR',
        aggfunc='mean',
        observed = True,
    )
    print(f"Genes found in cnv matrix: {len(df_cn['gene_id'].unique())}\n")

    common_genes = df_rna.index.intersection(df_cnv_pivot.index)
    print(f"Common Genes in expr and cn tables: {len(common_genes)}")

    df_rna = df_rna.loc[common_genes, :]
    df_cnv_pivot = df_cnv_pivot.loc[common_genes, :]

    # RNA Expression Variance
    logging.info(f"Computing RNA expression variance...")
    rna_variances = df_rna.var(axis=1)
    top_rna_series = rna_variances.nlargest(TOP_RNA_GENES)
    top_rna_genes_set = set(top_rna_series.index)
    logging.info(f"Selected top {len(top_rna_genes_set)} genes by RNA expression variance.\n")

    # Copy Number Variance (purifiedLogR)
    logging.info(f"Computing CN values variance...")
    #cn_variances = df_cn.groupby('gene_id', observed=True)['purifiedLogR'].var().dropna()
    cn_variances = df_cnv_pivot.var(axis=1)
    top_cn_series = cn_variances.nlargest(TOP_CN_GENES)
    top_cn_genes_set = set(top_cn_series.index)
    logging.info(f"Selected top {len(top_cn_genes_set)} genes by CN values variance.\n")

    ensg_to_symbol_map = df_cn.groupby('gene_id', observed=True)['symbol'].first().to_dict()

    selected_genes_union = top_rna_genes_set.union(top_cn_genes_set)
    overlap_count = len(top_rna_genes_set.intersection(top_cn_genes_set))

    logging.info("\n--- GENE SELECTION SUMMARY ---")
    logging.info(f"Top RNA genes count:        {len(top_rna_genes_set)}")
    logging.info(f"Top CN genes count:         {len(top_cn_genes_set)}")
    logging.info(f"Overlapping genes count:    {overlap_count}")
    logging.info(f"Unique Selected genes:   {len(selected_genes_union)}\n")

    logging.info(f"Adding Driver genes from '{driver_genes_filepath}'...")
    df_drivers = pd.read_csv(driver_genes_filepath, sep='\t')
    symbol_to_ensg = {v: k for k, v in ensg_to_symbol_map.items()}
    driver_genes_set = set(df_drivers['Symbol'].map(symbol_to_ensg).dropna())

    logging.info(f"Found {len(driver_genes_set)} unique driver gene symbols in file loaded.")
    selected_genes_union = top_rna_genes_set.union(top_cn_genes_set).union(driver_genes_set)
    logging.info(f"Driver genes count:          {len(driver_genes_set)}")
    new_drivers_added = len(driver_genes_set - top_rna_genes_set - top_cn_genes_set)
    logging.info(f"New Driver genes added:      {new_drivers_added}")
    logging.info(f"Total Unique Selected genes:  {len(selected_genes_union)}\n")

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
            #'in_top_rna': in_rna,
            'rna_variance': rna_variances.get(gene_id, np.nan),
            #'in_top_cn': in_cn,
            'cn_variance': cn_variances.get(gene_id, np.nan),
            'selection_reason': reason
        })

    df_selected = pd.DataFrame(rows).sort_values(by='rna_variance', ascending=False)
    df_selected.to_csv(output_selected_genes_filepath, sep='\t', index=False)
    logging.info(f"Saved selected genes to: '{output_selected_genes_filepath}' (Shape: {df_selected.shape})\n")


if __name__ == "__main__":
    select_genes()
    print(f"Finished in {time.time() - start_time}s")
    print("DONE\n")