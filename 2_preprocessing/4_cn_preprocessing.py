from pandas import read_csv
from os import path

import config_file_tree

dataset_dir = config_file_tree.DATASET_DIR
cnv_filepath = path.join(dataset_dir, "23Jun2025_combinedCnGenes.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
output_filepath = path.join(output_dir, "preprocessed_cn.tsv")


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

# removing Cell Line values
df_cnv_raw = df_cnv_raw[
    df_cnv_raw['sample'].str.contains("DNA", case=False, na=False)
    & ~df_cnv_raw['sample'].str.contains("CL", case=False, na=False)
].copy()

# todo keep only top genes by cnv variance? then union (set) with top rna

df_cnv_raw.to_csv(output_filepath, sep='\t', index=False)
