import logging
from os import path, makedirs

import numpy as np
from pandas import read_csv, DataFrame, concat

import config_file_tree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

""""" paths and parameters """""

rna_bulk_file = config_file_tree.RNA_EXTRACTED_FILE
string_aliases_filepath = path.join('..', 'ppi_string_files', '9606.protein.aliases.v12.0.txt')

dir_to_save = path.join("preprocessed_RNA_bulk")

top_genes_path = path.join(dir_to_save, "selected_top_genes.tsv")
filtered_matrix_path = path.join(dir_to_save, f"bulk_expr_top_genes_log2.tsv")


""""" utility functions """""

def get_protein_coding_ensg(filepath):
    """ Extract the Ensembl Gene ID (ENSG) of genes protein coding (the ones appearing in downloaded STRING aliases file). """

    df_aliases = read_csv(filepath, sep='\t', dtype={
        "#string_protein_id": "category",
        "alias": "str",
        "source": "category",
    }).dropna()

    print()
    print(df_aliases.info(memory_usage="deep"))
    print()

    ensg_mask = df_aliases['alias'].str.startswith('ENSG')
    df_ensg = df_aliases[ensg_mask].copy()
    df_ensg['ensg'] = df_ensg['alias'].str.split('.').str[0]

    prot_coding_ensg_set = set(df_ensg.loc[ensg_mask, 'alias'])  # extract ENSG cleaned

    print(f"Identified {len(prot_coding_ensg_set)} protein-coding genes from STRING.")


    # create file for gene IDs gene Symbols mapping:

    official_symbol_sources = [
        'Ensembl_HGNC_symbol',
        'BioMart_HUGO',
        'UniProt_GN_Name',
        'Ensembl_HGNC',
        'Ensembl_HGNC_alias_symbol',
        'Ensembl_EntrezGene',
        'Ensembl_WikiGene',
        'KEGG_NAME'
    ]

    symbol_mask = df_aliases['source'].isin(official_symbol_sources)
    df_symbols = df_aliases[symbol_mask].copy()

    # ENSGs related to Gene Symbols using proteins ID
    merged = df_ensg[['#string_protein_id', 'ensg']].merge(
        df_symbols[['#string_protein_id', 'alias']],
        on='#string_protein_id'
    ).drop_duplicates()

    # if some genes don't have any symbols matching, check other sources
    mapped_ensgs = set(merged['ensg'])
    unmapped_ensg = df_ensg[~df_ensg['ensg'].isin(mapped_ensgs)]

    df_symbols_backup = df_aliases[df_aliases['source'] == 'Ensembl_Clone_based_ensembl_gene']
    merged_2 = unmapped_ensg[['#string_protein_id', 'ensg']].merge(
        df_symbols_backup[['#string_protein_id', 'alias']],
        on='#string_protein_id'
    )[['ensg', 'alias']].drop_duplicates()

    merged = concat([merged, merged_2], ignore_index=True)

    merged['alias'] = merged['alias'].str.split('.').str[0]  # remove versioning (e.g. AC000357.1 -> AC000357)
    merged = merged.drop_duplicates().sort_values(by=['ensg', 'alias'])

    ensg_to_names = (
        merged.groupby('ensg')['alias']
        .agg(', '.join)
        .to_dict()
    )

    return prot_coding_ensg_set, ensg_to_names


def preprocess_hercules_rna_bulk(top_n_genes=5000):
    makedirs(dir_to_save, exist_ok=True)

    logging.info(f"Loading matrix (genes x samples) from {rna_bulk_file.split(path.sep)[-1]}...")
    df_bulk = read_csv(rna_bulk_file, sep="\t")

    gene_id_col = df_bulk.columns[0]
    sample_cols = list(df_bulk.columns[1:])

    # remove gene_id version number (es. ENSG00000141510.12 -> ENSG00000141510)
    df_bulk[gene_id_col] = df_bulk[gene_id_col].astype(str).str.split('.').str[0]

    # keep only protein_coding genes
    logging.info("Collecting protein coding genes...")
    protein_coding_gene_set, ensg_to_names_map = get_protein_coding_ensg(string_aliases_filepath)

    initial_gene_count = len(df_bulk)
    df_bulk = df_bulk[df_bulk[gene_id_col].isin(protein_coding_gene_set)]
    logging.info(f" --> Protein-coding filter: genes reduced from {initial_gene_count} to {len(df_bulk)}.")

    # if there are ambiguities (gene_ids duplicated), keep higher expression
    matrix_rna = df_bulk.groupby(gene_id_col)[sample_cols].max()

    logging.info("Computing log2(counts + 1)...")
    matrix_rna = np.log2(matrix_rna.fillna(0) + 1)

    logging.info(f"Selecting top {top_n_genes} genes by variance...\n")
    # transposed --> (Samples x Genes) so we can compute on columns, easier computation
    matrix_rna = matrix_rna.T
    gene_variances = matrix_rna.var()

    top_genes_var = gene_variances.nlargest(top_n_genes)
    matrix_rna = matrix_rna[top_genes_var.index]

    # saving as matrix (Genes x Samples)
    matrix_rna.T.to_csv(filtered_matrix_path, sep='\t')

    top_genes_df = DataFrame()
    top_genes_df['gene_id'] = top_genes_var.index
    top_genes_df['variance'] = top_genes_var.values
    top_genes_df['gene_name'] = [ensg_to_names_map.get(gene_id, 'N/A') for gene_id in top_genes_var.index]

    top_genes_df.to_csv(top_genes_path, sep='\t', index=False)
    logging.info(f"Selected top genes saved in: {top_genes_path}\n")


if __name__=="__main__":
    preprocess_hercules_rna_bulk()


print("DONE\n")