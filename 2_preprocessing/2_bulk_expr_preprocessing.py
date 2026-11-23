from os import path, makedirs
from pandas import read_csv, concat

import PCA
import config_file_tree


""""" paths and parameters """""

output_dir = config_file_tree.PREPROCESSED_DATA_DIR

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
rna_bulk_filepath = path.join(extracted_dir, "raw_bulk_expr.tsv")
metadata_filepath = path.join(output_dir, "preprocessed_samples_metadata.tsv")
string_aliases_filepath = config_file_tree.STRING_ALIASES_FILE

output_rna_bulk_filepath = path.join(output_dir, f"preprocessed_bulk_expr.tsv")


def get_protein_coding_ensg(filepath):
    """ Extract the Ensembl Gene ID (ENSG) of genes protein coding (the ones appearing in downloaded STRING aliases file). """

    df_aliases = read_csv(filepath, sep='\t', dtype={
        "#string_protein_id": "category",
        "alias": "str",
        "source": "category",
    }).dropna()

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


def filter_bulk_by_metadata():
    """ Keep only the sample columns in the bulk expression matrix that exist in preprocessed metadata. """
    print("Loading preprocessed metadata and raw bulk expression...")
    df_metadata = read_csv(metadata_filepath, sep='\t')
    df_bulk = read_csv(rna_bulk_filepath, sep="\t")
    gene_col = df_bulk.columns[0]

    valid_samples = df_metadata['sample_id'].tolist()
    print(f"Found {len(valid_samples)} samples in preprocessed metadata.")

    # Align bulk expression columns to preprocessed metadata sample IDs
    df_bulk_filtered = df_bulk[[gene_col] + valid_samples]
    print(f"Filtered bulk matrix shape: {df_bulk_filtered.shape}\n")

    return df_bulk_filtered


def filter_bulk_genes(df_bulk):
    """ Keep only protein coding genes and select top 5000 """
    gene_id_col = df_bulk.columns[0]

    # keep only protein_coding genes
    print("[FILTER PROTEIN-CODING] Collecting protein coding genes...")
    protein_coding_gene_set, ensg_to_names_map = get_protein_coding_ensg(string_aliases_filepath)

    initial_gene_count = len(df_bulk)
    df_bulk = df_bulk[df_bulk[gene_id_col].isin(protein_coding_gene_set)]
    print(f" --> genes reduced from {initial_gene_count} to {len(df_bulk)}.")

    # if there are ambiguities (gene_ids duplicated), keep higher expression
    sample_cols = list(df_bulk.columns[1:])
    matrix_rna = df_bulk.groupby(gene_id_col)[sample_cols].max()

    # saving as matrix (Genes x Samples)
    print(f"Saving preprocessed bulk expression matrix (Shape: {matrix_rna.shape}) to: {output_rna_bulk_filepath}\n")
    matrix_rna.to_csv(output_rna_bulk_filepath, sep='\t')


if __name__ == "__main__":
    bulk_filtered_df = filter_bulk_by_metadata()
    filter_bulk_genes(bulk_filtered_df)

    if config_file_tree.RUN_PCA:
        out_dir_pca = path.join(output_dir, "out_analysis")
        makedirs(out_dir_pca, exist_ok=True)

        PCA.run_pca_analysis(output_rna_bulk_filepath, metadata_filepath, out_dir_pca, 'Expr', 'cohort')
        PCA.run_pca_analysis(output_rna_bulk_filepath, metadata_filepath, out_dir_pca, 'Expr', 'tissue')
        PCA.run_pca_analysis(output_rna_bulk_filepath, metadata_filepath, out_dir_pca, 'Expr', 'tissue_type')
        PCA.run_pca_analysis(output_rna_bulk_filepath, metadata_filepath, out_dir_pca, 'Expr', 'is_fluid')
        PCA.run_pca_analysis(output_rna_bulk_filepath, metadata_filepath, out_dir_pca, 'Expr', 'phase_label')

    print("DONE\n")