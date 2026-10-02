import re

from pandas import read_csv, DataFrame
from os import path

import config_file_tree

dataset_dir = config_file_tree.DATASET_DIR
cnv_filepath = path.join(dataset_dir, "23Jun2025_combinedCnGenes.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
output_filepath = path.join(output_dir, "preprocessed_cn.tsv")


if __name__ == "__main__":
    print("Loading copy numbers data...")
    cols_to_keep = ["ID", "Gene", "type", "sample", "nMajor", "nMinor", "purifiedLogR", "breaksInGene"]
    # "nProbesCr", "nProbesAf", "logR", "baf", "nAraw", "nBraw", "purifiedBaf", "purifiedLoh", "CNstatus", "LOHstatus", "minPurifiedLogR", "maxPurifiedLogR"
    df_cnv_raw = read_csv(cnv_filepath, sep='\t', usecols=cols_to_keep,
                          dtype={
                  'type': 'category',
                  'sample': 'category',
                  'nMajor': 'float',
                  'nMinor': 'float',
                  'purifiedLogR': 'float',
                  'breaksInGene': 'float',
              })

    df_cnv_raw = df_cnv_raw[df_cnv_raw['type'] == 'protein_coding'].copy()

    df_cnv_raw = df_cnv_raw.drop(columns=['type'])

    df_cnv_raw = df_cnv_raw.rename(columns={"ID": "gene_id", "Gene": "symbol"})

    # removing Cell Line values
    df_cnv_raw = df_cnv_raw[
        df_cnv_raw['sample'].str.contains("DNA", case=False, na=False)
        & ~df_cnv_raw['sample'].str.contains("CL", case=False, na=False)
    ].copy()


    """"" multiple extractions deduplication (es. sample_DNA1, sample_DNA2) """""

    unique_samples = DataFrame({'sample': df_cnv_raw['sample'].unique()})
    unique_samples['tissue_base'] = unique_samples['sample'].apply(lambda x: re.split(r'_DNA', str(x))[0])

    # if terminates in DNA[number] or not (es. sequencing technique)
    unique_samples['is_canonical'] = unique_samples['sample'].apply(lambda x: bool(re.search(r'_DNA\d*$', str(x))))

    # keep canonical (remove duplicated samples names due to technology used)
    unique_samples = unique_samples[unique_samples['is_canonical']]

    num_duplicates = unique_samples.duplicated(subset=['tissue_base'], keep=False).sum()  # find duplicates
    print(f"CNV table with {len(unique_samples)} total samples.")
    print(
        f"Samples with re-extractions: {num_duplicates} -> keeping first version (DNA1)...")

    # re-ordering (es. 'DNA1' before 'DNA2') and keeping first
    unique_samples_valid = (
        unique_samples
        .sort_values(by='sample')
        .drop_duplicates(subset=['tissue_base'], keep='first')
    )
    valid_samples = set(unique_samples_valid['sample'])

    # filtering cn table
    df_cnv_raw = df_cnv_raw[df_cnv_raw['sample'].isin(valid_samples)].copy()

    print(f"Saving final CNV Dataframe with samples: {len(df_cnv_raw['sample'].unique())}")

    df_cnv_raw.to_csv(output_filepath, sep='\t', index=False)
