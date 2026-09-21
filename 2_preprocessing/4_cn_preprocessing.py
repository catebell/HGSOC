from pandas import read_csv
from os import path

import config_file_tree

dataset_dir = config_file_tree.DATASET_DIR
cnv_filepath = path.join(dataset_dir, "23Jun2025_combinedCnGenes.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
output_filepath = path.join(output_dir, "preprocessed_cn.tsv")


print("Loading copy numbers data...")
cols = ["ID", "Gene", "type", "sample", "nMajor", "nMinor", "purifiedLogR", "breaksInGene"]
# "nProbesCr", "nProbesAf", "logR", "baf", "nAraw", "nBraw", "purifiedBaf", "purifiedLoh", "CNstatus", "LOHstatus", "minPurifiedLogR", "maxPurifiedLogR"
df_cnv_raw = read_csv(cnv_filepath, sep='\t', usecols=cols,
          dtype={
              'type': 'category',
              'sample': 'category',
              'nMajor': 'float',
              'nMinor': 'float',
              'purifiedLogR': 'float',
              'breaksInGene': 'float',
          })
df_cnv_raw.info(memory_usage='deep')
df_cnv_raw = df_cnv_raw[df_cnv_raw['type'] == 'protein_coding'].copy()  # df_cnv_raw.query("type == 'protein_coding'").copy()
df_cnv_raw = df_cnv_raw.drop(columns=['type'])

# todo keep only top genes?

df_cnv_raw.to_csv(output_filepath, sep='\t', index=False)
