import os
import numpy as np
import pandas as pd
import pyranges as pr
from tqdm import tqdm

import config_file_tree

"""
Extract and map PURPLE genomic segment CNV data to gene-level features.
Processes sample-by-sample to optimize RAM usage.
Computes log2CopyNumber as the purity-corrected log-ratio replacement for GAT graph inputs.
"""

dataset_dir = config_file_tree.DATASET_DIR

purple_filepath = config_file_tree.ORIGINAL_PURPLE_CN_FILE
ascat_ref_filepath = config_file_tree.ORIGINAL_ASCAT_CN_FILE  # to retrieve gene coordinates

output_filepath = os.path.join(config_file_tree.EXTRACTED_DATA_DIR, "raw_cn_data.tsv")


def load_gene_reference(ascat_path):
    """ Extract unique gene annotations and genomic coordinates from reference file. """
    print(f"Loading gene reference coordinates from: {ascat_path}")
    ref_cols = ["ID", "Gene", "chr", "start", "end", "strand", "band", "type"]

    df_ref = pd.read_csv(ascat_path, sep='\t', usecols=ref_cols,
                         dtype={
                             'ID': 'category',
                             'Gene': 'category',
                             'chr': 'category',
                             'band': 'category',
                             'type': 'category',
                         })
    df_genes = df_ref.drop_duplicates(subset=['ID', 'Gene']).copy()

    # Standardize column names for PyRanges
    df_genes['Chromosome'] = df_genes['chr'].astype(str)
    df_genes['Chromosome'] = df_genes['Chromosome'].apply(lambda c: c if c.startswith('chr') else f"chr{c}")
    df_genes['Start'] = df_genes['start'].astype(int)
    df_genes['End'] = df_genes['end'].astype(int)

    print(f"Extracted {len(df_genes)} unique genes for annotation.")
    return df_genes


def process_sample_segments(df_sample_purple, pr_genes, df_genes):
    """ Map PURPLE segments for a single sample to genes using PyRanges and compute overlap-weighted metrics. """

    # Imputation of missing values in segment metrics before weighting
    df_sample_purple = df_sample_purple.copy()
    df_sample_purple['copyNumber'] = df_sample_purple['copyNumber'].fillna(2.0)
    df_sample_purple['nMajor'] = df_sample_purple['nMajor'].fillna(1.0)
    df_sample_purple['nMinor'] = df_sample_purple['nMinor'].fillna(1.0)
    df_sample_purple['logR'] = df_sample_purple['logR'].fillna(0.0)
    df_sample_purple['Loh'] = df_sample_purple['Loh'].fillna(0.0)
    df_sample_purple['baf'] = df_sample_purple['baf'].fillna(0.0)

    pr_purple = pr.PyRanges(
        df_sample_purple[
            ['Chromosome', 'Start', 'End', 'sample', 'copyNumber', 'nMajor', 'nMinor', 'logR', 'Loh', 'baf']]
    )

    # Spatial join for a single sample
    joined_pr = pr_genes.join(pr_purple, suffix='_purple')
    df_joined = joined_pr.df

    if df_joined.empty:
        return pd.DataFrame()

    # Calculate exact overlap boundaries and length
    df_joined['overlap_start'] = df_joined[['Start', 'Start_purple']].max(axis=1)
    df_joined['overlap_end'] = df_joined[['End', 'End_purple']].min(axis=1)
    df_joined['overlap_len'] = (df_joined['overlap_end'] - df_joined['overlap_start']).clip(lower=1)

    # Weighted metrics for multi-segment overlaps
    df_joined['weighted_copyNumber'] = df_joined['copyNumber'] * df_joined['overlap_len']
    df_joined['weighted_nMajor'] = df_joined['nMajor'] * df_joined['overlap_len']
    df_joined['weighted_nMinor'] = df_joined['nMinor'] * df_joined['overlap_len']
    df_joined['weighted_logR'] = df_joined['logR'] * df_joined['overlap_len']
    df_joined['weighted_Loh'] = df_joined['Loh'] * df_joined['overlap_len']
    df_joined['weighted_baf'] = df_joined['baf'] * df_joined['overlap_len']

    # Group by Sample and Gene ID
    grouped = df_joined.groupby(['sample', 'ID'], as_index=False).agg(
        total_overlap=('overlap_len', 'sum'),
        sum_copyNumber=('weighted_copyNumber', 'sum'),
        sum_nMajor=('weighted_nMajor', 'sum'),
        sum_nMinor=('weighted_nMinor', 'sum'),
        sum_logR=('weighted_logR', 'sum'),
        sum_Loh=('weighted_Loh', 'sum'),
        sum_baf=('weighted_baf', 'sum'),
        num_segments=('overlap_len', 'count')
    )

    # Compute final weighted averages
    grouped['copyNumber'] = grouped['sum_copyNumber'] / grouped['total_overlap']
    grouped['nMajor'] = grouped['sum_nMajor'] / grouped['total_overlap']
    grouped['nMinor'] = grouped['sum_nMinor'] / grouped['total_overlap']
    grouped['logR'] = grouped['sum_logR'] / grouped['total_overlap']
    grouped['Loh'] = grouped['sum_Loh'] / grouped['total_overlap']
    grouped['baf'] = grouped['sum_baf'] / grouped['total_overlap']
    grouped['breaksInGene'] = (grouped['num_segments'] - 1).clip(lower=0)

    # Derivation of log2CopyNumber (purity-corrected log2 ratio)
    grouped['log2CopyNumber'] = np.log2(np.maximum(grouped['copyNumber'], 0.001) / 2.0)

    # Merge gene metadata back
    df_final = grouped[['sample', 'ID', 'copyNumber', 'log2CopyNumber', 'nMajor', 'nMinor', 'logR', 'Loh', 'baf',
                        'breaksInGene']].merge(
        df_genes[['ID', 'Gene', 'chr', 'start', 'end', 'strand', 'band', 'type']],
        on='ID',
        how='left'
    )

    # Output column layout
    useful_columns = [
        "ID", "Gene", "chr", "start", "end", "strand", "band", "type", "sample",
        "copyNumber", "log2CopyNumber", "nMajor", "nMinor", "logR", "Loh", "baf", "breaksInGene"
    ]

    return df_final[useful_columns]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    df_genes_ref = load_gene_reference(ascat_ref_filepath)
    pr_genes_ref = pr.PyRanges(
        df_genes_ref[['Chromosome', 'Start', 'End', 'ID', 'Gene', 'type', 'chr', 'start', 'end', 'strand', 'band']]
    )

    print(f"\nLoading PURPLE segmentation data from: {purple_filepath}")
    cols = ['sample', 'patient', 'chromosome', 'start', 'end', 'copyNumber', 'baf', 'minorAlleleCopyNumber',
            'majorAlleleCopyNumber', 'logR', 'Loh']

    df_purple_raw = pd.read_csv(purple_filepath, sep='\t', usecols=cols, dtype={
        'sample': 'category',
        'patient': 'category',
        'chromosome': 'category',
    })

    # Normalize column names from PURPLE
    column_mapping = {}
    for col in df_purple_raw.columns:
        clow = col.lower()
        if 'chrom' in clow:
            column_mapping[col] = 'Chromosome'
        elif clow == 'start':
            column_mapping[col] = 'Start'
        elif clow == 'end':
            column_mapping[col] = 'End'
        elif 'major' in clow:
            column_mapping[col] = 'nMajor'
        elif 'minor' in clow:
            column_mapping[col] = 'nMinor'
        elif clow == 'logr':
            column_mapping[col] = 'logR'

    df_purple_raw = df_purple_raw.rename(columns=column_mapping)
    df_purple_raw['Chromosome'] = df_purple_raw['Chromosome'].astype(str)
    df_purple_raw['Chromosome'] = df_purple_raw['Chromosome'].apply(lambda c: c if c.startswith('chr') else f"chr{c}")

    unique_samples = df_purple_raw['sample'].unique()
    print(f"Found {len(unique_samples)} unique samples in PURPLE file.")
    print("Processing sample-by-sample to optimize RAM usage...\n")

    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    write_header = True
    total_records = 0

    for sample_name, df_sample in tqdm(df_purple_raw.groupby('sample'), desc="Processing samples"):
        df_sample_mapped = process_sample_segments(df_sample, pr_genes_ref, df_genes_ref)
        if df_sample_mapped.empty:
            continue

        df_sample_mapped.to_csv(output_filepath, sep='\t', index=False, mode='a', header=write_header)
        write_header = False
        total_records += len(df_sample_mapped)

    print(f"\nSaved generated raw CN data to: {output_filepath}")
    print(f"Done! Successfully processed and saved {total_records} gene-sample records.")