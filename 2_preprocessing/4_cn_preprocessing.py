import re

from pandas import read_csv, DataFrame
from os import path

import config_file_tree

cnv_filepath = path.join(config_file_tree.EXTRACTED_DATA_DIR, "raw_cn_data.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
expr_filepath = path.join(output_dir, 'preprocessed_bulk_expr.tsv')
output_filepath = path.join(output_dir, "preprocessed_cn.tsv")


def rna_cn_samples_mapping(rna_samples_ids, cn_samples_ids):
    """ Maps every RNA sample_id to the respective CN sample (es. 'H092_r1Asc1_DNA1' -> 'H092_r1Asc1') """

    # map: CN sample_id -> tissue (es. 'H092_r1Asc1_DNA1' -> 'H092_r1Asc1')
    cn_sample_map = {cn_s: str(cn_s).split('_DNA')[0] for cn_s in cn_samples_ids}

    mapping_rows = []
    for rna_s in rna_samples_ids:
        # find CN samples with prefix = prefix in rna samples
        matches = [
            cn_s for cn_s, tissue_prefix in cn_sample_map.items()
            if rna_s.startswith(tissue_prefix + "_") or rna_s == tissue_prefix
        ]

        if len(matches) == 1:
            mapping_rows.append({'sample_id': rna_s, 'cn_sample': matches[0], 'status': 'matched'})
        elif len(matches) > 1:
            mapping_rows.append({'sample_id': rna_s, 'cn_sample': None, 'status': f'ambiguous ({matches})'})  # seems 0
        else:
            mapping_rows.append({'sample_id': rna_s, 'cn_sample': None, 'status': 'unmapped'})

    df_map = DataFrame(mapping_rows)

    print("\n--- SAMPLES MAPPING RECAP ---")
    print(df_map['status'].value_counts())
    print("\n")

    # CN samples mapped to many RNA samples (1-a-N)
    matched_df = df_map[df_map['status'] == 'matched']
    cn_counts = matched_df['cn_sample'].value_counts()
    multi_mapped_cn = cn_counts[cn_counts > 1]

    print(f"--- CN SAMPLES MAPPED TO MULTIPLE RNA SAMPLES: {len(multi_mapped_cn)} ---")
    for cn_s, count in multi_mapped_cn.items():
        associated_rna = matched_df[matched_df['cn_sample'] == cn_s]['sample_id'].tolist()
        print(f"\nCN sample: '{cn_s}' -> Mapped to {count} RNA samples:")
        for rna in associated_rna:
            print(f"   - {rna}")

    unmapped_rna = df_map[df_map['status'] == 'unmapped']['sample_id'].tolist()
    if unmapped_rna:
        print(f"\n--- UNMATCHED: {len(unmapped_rna)} RNA samples with no CN match ---")
        print(unmapped_rna[:10], "... (first 10)" if len(unmapped_rna) > 10 else "")

    return matched_df[['sample_id', 'cn_sample']]


if __name__ == "__main__":
    print("Loading copy numbers data...")
    cols_to_keep = [
        "ID", "Gene", "type", "sample",
        "copyNumber", "log2CopyNumber", "nMajor", "nMinor", 'logR', "Loh", "baf", "breaksInGene"
    ]
    #cols_to_keep = ["ID", "Gene", "type", "sample", "nMajor", "nMinor", "purifiedLogR", "breaksInGene"]
    # "nProbesCr", "nProbesAf", "logR", "baf", "nAraw", "nBraw", "purifiedBaf", "purifiedLoh", "CNstatus", "LOHstatus", "minPurifiedLogR", "maxPurifiedLogR"
    df_cnv_raw = read_csv(cnv_filepath, sep='\t', usecols=cols_to_keep,
                          dtype={
                              'ID': 'category',
                              'Gene': 'category',
                              'type': 'category',
                              'sample': 'category',
                          })

    df_cnv_raw = df_cnv_raw[df_cnv_raw['type'] == 'protein_coding'].copy()
    df_cnv_raw = df_cnv_raw.drop(columns=['type'])
    df_cnv_raw = df_cnv_raw.rename(columns={"ID": "gene_id", "Gene": "symbol"})

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

    # filtering CN table to keep valid samples
    df_cnv_raw = df_cnv_raw[df_cnv_raw['sample'].isin(valid_samples)].copy()

    # alignment with RNA sample_ids
    print("\nLoading RNA expression matrix sample_ids to align samples...")
    df_expr_headers = read_csv(expr_filepath, sep="\t", index_col=0, nrows=1)  # sample ids are in header
    rna_sample_ids = df_expr_headers.columns.tolist()
    cn_sample_ids = df_cnv_raw['sample'].unique().tolist()

    df_matched = rna_cn_samples_mapping(rna_sample_ids, cn_sample_ids)

    #assign 'sample_id' and discard CN samples with no association
    df_cnv_mapped = df_cnv_raw.merge(df_matched, left_on='sample', right_on='cn_sample', how='inner')
    df_cnv_mapped = df_cnv_mapped.drop(columns=['cn_sample'])

    #cols_order = ['gene_id', 'symbol', 'sample', 'sample_id', 'purifiedLogR', 'nMajor', 'nMinor', 'breaksInGene']
    cols_order = [
        'gene_id', 'symbol', 'sample', 'sample_id',
        'copyNumber', 'log2CopyNumber', 'nMajor', 'nMinor', 'logR', 'Loh', 'baf', 'breaksInGene'
    ]
    df_cnv_mapped = df_cnv_mapped[cols_order]

    print(f"\nSaving final preprocessed CN Dataframe...")
    print(f"    (Total rows: {len(df_cnv_mapped)})")
    print(f"    (Unique sample ids mapped: {df_cnv_mapped['sample_id'].nunique()})")

    df_cnv_mapped.to_csv(output_filepath, sep='\t', index=False)

    print("\nDONE!\n")
