import logging
from os import path
from pandas import read_csv
from tqdm import tqdm

import config_file_tree


""""" paths and parameters """""

rna_bulk_raw_filepath = config_file_tree.RNA_BULK_RAW_FILE
deconv_w_filepath = config_file_tree.DECONVOLUTION_W_FILE

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR  # is also output dir

metadata_path = path.join(extracted_dir, "samples_metadata.tsv")

COHORT_THRESHOLD = 1  # % of minimum samples per cohort wrt the total num of samples
EOC_THRESHOLD = 0.30

def filter_bulk():
    print("Loading files...")

    df_metadata = read_csv(metadata_path, sep='\t', usecols=['sample_id', 'cohort', 'sample_class'])

    df_bulk = read_csv(rna_bulk_raw_filepath, sep="\t")
    #df_bulk.info(memory_usage='deep')
    gene_col = df_bulk.columns[0]

    df_w = read_csv(deconv_w_filepath, sep='\t', dtype={
        'EOC': 'float32', 'Fibroblast': 'float32', 'Immune': 'float32', 'Unknown': 'float32'
    })

    logging.info("[FILTERING] Removing Cell lines, only BulkTissue kept...")
    df_metadata = df_metadata[df_metadata['sample_class'] == 'BulkTissue']
    df_bulk = df_bulk[[gene_col] + df_metadata['sample_id'].to_list()]

    total_samples = len(df_metadata)
    print(f"\nFound {total_samples} BulkTissue samples")

    logging.info(f"\n[FILTERING] Applying Tumor Purity filter (EOC > {EOC_THRESHOLD * 100:.0f}%)...")
    sample_col = df_w.columns[0]
    eoc_valid_samples = set(df_w.loc[df_w['EOC'] > EOC_THRESHOLD, :][sample_col])
    df_metadata = df_metadata[df_metadata['sample_id'].isin(eoc_valid_samples)].copy()
    df_bulk = df_bulk[[gene_col] + df_metadata['sample_id'].to_list()]

    total_samples = len(df_metadata)
    print(f"\nKept {total_samples} samples having EOC > {EOC_THRESHOLD}")


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
        print(f"  - Tissues:  {n_tissues}")


    """"" filter out cohorts with less than 10% samples of the dataset """

    cohorts_to_keep = [s['cohort'] for s in cohort_stats if s['percentage'] >= COHORT_THRESHOLD]
    cohorts_to_remove = [s['cohort'] for s in cohort_stats if s['percentage'] < COHORT_THRESHOLD]

    print(f"\n[FILTERING] Kept Cohorts (>= {COHORT_THRESHOLD}%): {cohorts_to_keep}")
    if cohorts_to_remove:
        print(f"[FILTERING] Removed Cohorts (< {COHORT_THRESHOLD}%): {cohorts_to_remove}")

    # filtering metadata
    df_metadata_filtered = df_metadata[df_metadata['cohort'].isin(cohorts_to_keep)].copy()
    valid_samples = df_metadata_filtered['sample_id'].tolist()

    print(f"\nKeeping {len(valid_samples)} samples...")
    df_bulk_filtered = df_bulk[[gene_col] + valid_samples]

    output_bulk_file = path.join(extracted_dir, "bulk_expr_filtered.tsv")
    logging.info(f"\nSaving filtered bulk matrix: '{output_bulk_file}' (Shape: {df_bulk_filtered.shape})...")
    df_bulk_filtered.to_csv(output_bulk_file, sep="\t", index=False)

    #output_meta_file = path.join(extracted_dir, "samples_metadata_filtered.tsv")
    #df_metadata_filtered.to_csv(output_meta_file, sep="\t", index=False)
    #print(f"Saved filtered metadata: '{output_meta_file}' (Shape: {df_metadata_filtered.shape})")


if __name__ == "__main__":
    filter_bulk()
    print("\nDONE!")