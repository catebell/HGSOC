from os import path, makedirs
from pandas import read_csv
from tqdm import tqdm

import config_file_tree


""""" paths and parameters """""

dataset_dir = config_file_tree.DATASET_DIR
rna_bulk_raw_filepath = config_file_tree.RNA_BULK_RAW_FILE
rna_bulk_raw_df = read_csv(rna_bulk_raw_filepath, sep='\t')

metadata_path = path.join(dataset_dir, "samples_metadata.tsv")
dir_to_save = path.join(dataset_dir, "extracted")


def split_bulk_by_cohort():
    makedirs(dir_to_save, exist_ok=True)

    print("Loading files...")

    df_metadata = read_csv(metadata_path, sep='\t', usecols=['sample_id', 'cohort', 'sample_class'])
    #print(metadata_filepath.info(memory_usage='deep'))

    df_bulk = read_csv(rna_bulk_raw_filepath, sep="\t")
    gene_col = df_bulk.columns[0]

    print("Removing Cell lines, only BulkTissue kept...")
    df_metadata = df_metadata[df_metadata['sample_class'] == 'BulkTissue']

    print("Separating cohorts...")

    for cohort_name, group in tqdm(df_metadata.groupby('cohort')):
        cohort_samples = group['sample_id'].tolist()
        valid_samples = [col for col in cohort_samples if col in df_bulk.columns]

        if not valid_samples:
            print(f"[SKIP] No sample found for cohort: {cohort_name}")
            continue

        df_cohort = df_bulk[[gene_col] + valid_samples]  # sub matrix extraction

        output_file = path.join(dir_to_save, f"bulk_expr_cohort_{cohort_name}.tsv")
        df_cohort.to_csv(output_file, sep="\t", index=False)

        print(f"\nCohort '{cohort_name}': {len(valid_samples)} samples, saved in '{output_file}' ({df_cohort.shape})\n")


if __name__ == "__main__":
    split_bulk_by_cohort()
    print("\nDONE!")