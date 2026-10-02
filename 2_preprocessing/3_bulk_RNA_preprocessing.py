from os import path, makedirs

from pandas import read_csv, DataFrame, concat
from tqdm import tqdm

import PCA
import config_file_tree


""""" paths and parameters """""

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
makedirs(output_dir, exist_ok=True)

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
rna_bulk_filepath = path.join(extracted_dir, "raw_bulk_expr.tsv")
metadata_filepath = path.join(output_dir, "preprocessed_samples_metadata.tsv")

deconv_w_filepath = config_file_tree.DECONVOLUTION_W_FILE  # for filtering
string_aliases_filepath = config_file_tree.STRING_ALIASES_FILE

COHORT_THRESHOLD = 1  # % of minimum samples per cohort wrt the total num of samples
EOC_THRESHOLD = 0.30  # minimum tumor purity

output_rna_bulk_filepath = path.join(output_dir, f"preprocessed_bulk_expr.tsv")
output_metadata_filepath = path.join(output_dir, f"preprocessed_samples_metadata.tsv")


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


def filter_bulk_and_metadata():
    print("Loading files...")
    df_w = read_csv(deconv_w_filepath, sep='\t', usecols=[0, 1], dtype={'EOC': 'float32'})

    df_metadata = read_csv(metadata_filepath, sep='\t')
    df_bulk = read_csv(rna_bulk_filepath, sep="\t")
    gene_col = df_bulk.columns[0]

    total_samples = len(df_metadata)
    print(f"Found {total_samples} samples in metadata (and expression table).\n")

    print(f"[FILTERING PURITY] Applying Tumor Purity filter (EOC > {EOC_THRESHOLD * 100:.0f}%)...")
    sample_col = df_w.columns[0]
    eoc_valid_samples = set(df_w.loc[df_w['EOC'] > EOC_THRESHOLD, :][sample_col])
    df_metadata = df_metadata[df_metadata['sample_id'].isin(eoc_valid_samples)].copy()
    df_bulk = df_bulk[[gene_col] + df_metadata['sample_id'].to_list()]

    total_samples = len(df_metadata)
    print(f"Kept {total_samples} samples having EOC > {EOC_THRESHOLD}\n")


    print("Cohorts stats:")
    cohort_stats = []

    for cohort_name, group in tqdm(df_metadata.groupby('cohort')):
        n_samples = len(group)
        percentage = (n_samples / total_samples) * 100
        n_patients = group['patient_id'].nunique() if 'patient_id' in group.columns else 0
        n_tissues = group['tissue'].nunique() if 'tissue' in group.columns else 0

        cohort_stats.append({
            'cohort': cohort_name,
            'n_samples': n_samples,
            'percentage': percentage,
            'n_patients': n_patients,
            'n_tissues': n_tissues
        })

        print(f"Cohort '{cohort_name}':")
        print(f"  - Samples:  {n_samples} ({percentage:.2f}%)")
        print(f"  - Patients: {n_patients}")
        print(f"  - Tissues:  {n_tissues}\n")


    """"" filter out cohorts with less than 10% samples of the dataset """

    cohorts_to_keep = [s['cohort'] for s in cohort_stats if s['percentage'] >= COHORT_THRESHOLD]
    cohorts_to_remove = [s['cohort'] for s in cohort_stats if s['percentage'] < COHORT_THRESHOLD]

    print(f"[FILTERING COHORT] Kept Cohorts (>= {COHORT_THRESHOLD}%): {cohorts_to_keep}\n")
    if cohorts_to_remove:
        print(f"[FILTERING COHORT] Removed Cohorts (< {COHORT_THRESHOLD}%): {cohorts_to_remove}\n")

    # filtering metadata
    df_metadata_filtered = df_metadata[df_metadata['cohort'].isin(cohorts_to_keep)].copy()
    valid_samples = df_metadata_filtered['sample_id'].tolist()

    df_bulk_filtered = df_bulk[[gene_col] + valid_samples]
    print(f"Filtered bulk matrix (Shape: {df_bulk_filtered.shape})...\n")

    # deduplication of multiple RNA extraction
    dup_mask = df_metadata_filtered.duplicated('sample', keep=False)
    num_duplicates = dup_mask.sum()
    print(f"[DEDUPLICATING SAMPLES] Found {num_duplicates} duplicated samples cols (multiple extractions).")

    # re-ordering by 'extraction' (es. 'RNA1' before 'RNA2', 'RNA3', ...) and keeping first
    df_metadata_filtered = (
        df_metadata_filtered
        .sort_values(by='extraction')
        .drop_duplicates(subset=['sample'], keep='first')
        .sort_index()
    )

    # aligning bulk and metadata
    valid_samples = df_metadata_filtered['sample_id'].tolist()
    df_bulk_filtered = df_bulk[[gene_col] + valid_samples]

    df_metadata_filtered.to_csv(output_metadata_filepath, sep="\t", index=False)
    print(f"Saved filtered metadata: '{output_metadata_filepath}' (Shape: {df_metadata_filtered.shape})\n")

    return df_bulk_filtered, df_metadata_filtered


def filter_bulk_genes(df_bulk, df_metadata, top_n_genes=5000):
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
    matrix_rna.to_csv(output_rna_bulk_filepath, sep='\t')
    print(f"Saved preprocessed bulk expression matrix to: {output_rna_bulk_filepath} (Shape: {matrix_rna.shape})\n")


if __name__ == "__main__":
    bulk_filtered_df, metadata_filtered = filter_bulk_and_metadata()
    filter_bulk_genes(bulk_filtered_df, metadata_filtered)

    if config_file_tree.RUN_PCA:
        out_dir_pca = path.join(output_dir, "out_analysis")
        makedirs(out_dir_pca, exist_ok=True)

        PCA.run_pca_analysis(output_rna_bulk_filepath, output_metadata_filepath, out_dir_pca, 'cohort')
        PCA.run_pca_analysis(output_rna_bulk_filepath, output_metadata_filepath, out_dir_pca, 'tissue')
        PCA.run_pca_analysis(output_rna_bulk_filepath, output_metadata_filepath, out_dir_pca, 'phase_label')

    print("DONE\n")