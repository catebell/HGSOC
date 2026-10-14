import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

import config_file_tree

"""
Split based on metadata and patient_id + metadata and clinical processing
"""

""""" paths and parameters """""

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
graph_dir = config_file_tree.PROCESSED_SAMPLES_DIR

metadata_filepath = os.path.join(graph_dir, 'associated_metadata.tsv')
clinical_filepath = os.path.join(preprocessed_dir, "preprocessed_clinical_data.tsv")

output_metadata_filepath = "metadata_for_training.tsv"
output_clinical_filepath = "clinical_data_for_training.tsv"

target_col_metadata = config_file_tree.SAMPLE_TARGET_COL  # task label to stratify on
target_col_clinical = config_file_tree.PATIENT_TARGET_COL
task = config_file_tree.TASK

time_col = config_file_tree.TIME_COL if task == 'survival' else ""

# 70% Train, 15% Val, 15% Test
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 7

MIN_PATIENTS_PER_CLASS = 5


def clean_feature_cols(df_metadata, df_clinical):
    """ Keep only cols needed for sample metadata. """

    # SAMPLES FEATURES

    cols_to_keep_metadata = ['sample_id', 'patient_id']

    if target_col_metadata in df_metadata.columns:
        cols_to_keep_metadata.append(target_col_metadata)

    if config_file_tree.USE_SAMPLES_METADATA:
        df_metadata_filtered = df_metadata[cols_to_keep_metadata + config_file_tree.METADATA_FEATURES].copy()
    else:
        df_metadata_filtered = df_metadata[cols_to_keep_metadata].copy()

    print(f"Metadata cleaned, kept cols: {df_metadata_filtered.columns.to_list()}")

    # PATIENT FEATURES

    cols_to_keep_clinical = ['patient_id']

    if target_col_clinical in df_clinical.columns:
        cols_to_keep_clinical.append(target_col_clinical)

    if config_file_tree.USE_CLINICAL_DATA:
        df_clinical_filtered = df_clinical[cols_to_keep_clinical + config_file_tree.CLINICAL_FEATURES].copy()
    else:
        df_clinical_filtered = df_clinical[cols_to_keep_clinical].copy()

    print(f"Clinical data cleaned, kept cols: {df_clinical_filtered.columns.to_list()}\n")

    return df_metadata_filtered, df_clinical_filtered


def clean_and_filter_target(df_metadata, df_clinical):
    """ Exclude target NaNs and rare classes, saving remaining samples. """

    # METADATA

    print(f"\nDeduplicated samples in mapping: {len(df_metadata)} ({df_metadata['patient_id'].nunique()} patients)")

    initial_len = len(df_metadata)
    df_clean = df_metadata.dropna(subset=[target_col_metadata]).copy()
    if len(df_clean) < initial_len:
        print(f"[FILTER-METADATA] Dropped {initial_len - len(df_clean)} samples with NaN in target '{target_col_metadata}'.")

    if task == 'survival':
        if time_col and time_col in df_clean.columns:
            n_before_time = len(df_clean)
            df_clean = df_clean[df_clean[time_col] > 0].dropna(subset=[time_col]).copy()
            print(f"[FILTER] Removed {n_before_time - len(df_clean)} samples with invalid/missing time col '{time_col}'.")
        print(f"\n--- Sample Task: SURVIVAL ---\n")
        print(f"Event Col: '{target_col_metadata}' | Time Col: '{time_col}'")
        print("Event distribution between samples:\n", df_clean[target_col_metadata].value_counts(normalize=True))

    elif task == 'classification':
        print(f"\n--- Sample Task: CLASSIFICATION ---\n")
        print(f"Sample Target Label: '{target_col_metadata}'")
        print("\n(Sample-level) Classes distribution:\n", df_clean[target_col_metadata].value_counts(normalize=True))


    """"" Removing rare classes (by patient or sample) """""

    all_classes = df_clean[target_col_metadata].value_counts()

    print(f"\n(Patient-level) rare sample classes filtering...")
    df_patients = df_clean[['patient_id', target_col_metadata]].drop_duplicates(subset=['patient_id'], keep='first')
    patients_per_class = df_patients[target_col_metadata].value_counts()
    valid_classes = patients_per_class[patients_per_class >= MIN_PATIENTS_PER_CLASS].index
    removed_classes = set(all_classes.index) - set(valid_classes)

    if removed_classes:
        print(f"[FILTER-METADATA] Removed sample classes with < {MIN_PATIENTS_PER_CLASS} unique patients: {removed_classes}")
        df_clean = df_clean[df_clean[target_col_metadata].isin(valid_classes)].copy()

    # CLINICAL

    if config_file_tree.PATIENTS_LEVEL_TRAIN:
        n_before_drop = len(df_clinical)
        df_clinical = df_clinical.dropna(subset=[target_col_clinical]).copy()
        if len(df_clinical) < n_before_drop:
            print(
                f"[FILTER-CLINICAL] Dropped {n_before_drop - len(df_clinical)} clinical records with NaN in patient target '{target_col_clinical}'.")

        print(f"\nPatient Target Label: '{target_col_clinical}'")
        print("Classes distribution (Patient level):\n", df_clinical[target_col_clinical].value_counts(normalize=True))

        clinical_class_counts = df_clinical[target_col_clinical].value_counts()
        valid_clinical_classes = clinical_class_counts[clinical_class_counts >= MIN_PATIENTS_PER_CLASS].index
        removed_clinical_classes = set(clinical_class_counts.index) - set(valid_clinical_classes)

        if removed_clinical_classes:
            print(
                f"[FILTER-CLINICAL] Removed patient target classes with < {MIN_PATIENTS_PER_CLASS} patients: {removed_clinical_classes}\n")
            df_clinical = df_clinical[df_clinical[target_col_clinical].isin(valid_clinical_classes)].copy()
            print("Updated Patient Classes Distribution after rare filtering (Counts):\n",
                  df_clinical[target_col_clinical].value_counts())
            print("Updated Patient Classes Distribution after rare filtering (%):\n",
                  df_clinical[target_col_clinical].value_counts(normalize=True).round(4))


    # keep only patients in clinical present in metadata cleaned
    valid_patients = set(df_clean['patient_id'].astype(str).str.strip().str.upper().unique())
    n_patients = len(df_clinical)
    df_clinical = df_clinical[df_clinical['patient_id'].isin(valid_patients)].copy()
    print(f"\n[FILTER-CLINICAL] Removed {n_patients - len(df_clinical)} patients with no samples in processed metadata.")

    # saving to files
    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    print(f"\nSaved processed metadata ({df_clean.shape}) to {output_metadata_filepath}")

    df_clinical.to_csv(output_clinical_filepath, sep='\t', index=False)
    print(f"Saved processed clinical data ({df_clinical.shape}) to {output_clinical_filepath}\n")

    return df_clean, df_clinical


def perform_split_and_save(df_clean, df_clinical):
    """ Split based on metadata, stratifying by patient or sample. """

    ratio_val_test = VAL_SIZE + TEST_SIZE  # (0.15 + 0.15 = 0.30 --> 30%)
    ratio_test = TEST_SIZE / ratio_val_test  # 0.50

    print(f"\n[PATIENT-LEVEL stratified split]")
    df_patients = df_clean[['patient_id', target_col_metadata]].drop_duplicates(subset=['patient_id'], keep='first')

    # TODO? add bool to enable or not. Se enabled serve fare check per escludere quelle troppo rare?
    '''# merge for stratification on both targets (sample and patient)
    df_patients = df_patients.merge(
        df_clinical[['patient_id', target_col_clinical]],
        on='patient_id',
        how='inner'
    )
    df_patients['composite_target'] = (
            df_patients[target_col_metadata].astype(str) + "_" +
            df_patients[target_col_clinical].astype(str)
    )

    print("Composite Target (by patient) distribution:\n", df_patients['composite_target'].value_counts())'''

    # split train / val+test
    train_pts_df, val_test_pts_df = train_test_split(
        df_patients,
        test_size=ratio_val_test,
        random_state=RANDOM_SEED,
        stratify=df_patients[target_col_metadata]
        #stratify=df_patients['composite_target']
    )

    # split val / test (remaining patients)
    val_pts_df, test_pts_df = train_test_split(
        val_test_pts_df,
        test_size=ratio_test,
        random_state=RANDOM_SEED,
        stratify=val_test_pts_df[target_col_metadata]
        #stratify=val_test_pts_df['composite_target']
    )

    # patient mapping to retrieve related tissues
    df_train = df_clean[df_clean['patient_id'].isin(train_pts_df['patient_id'])].copy()
    df_val   = df_clean[df_clean['patient_id'].isin(val_pts_df['patient_id'])].copy()
    df_test  = df_clean[df_clean['patient_id'].isin(test_pts_df['patient_id'])].copy()

    df_train_val = pd.DataFrame(pd.concat([df_train, df_val]))

    df_train_val.to_csv("samples_train_val.csv", index=False)
    df_train.to_csv("samples_train.csv", index=False)
    df_val.to_csv("samples_val.csv", index=False)
    df_test.to_csv("samples_test.csv", index=False)

    print("Split Summary:")
    total_valid_samples = len(df_clean)
    total_valid_patients = df_clean['patient_id'].nunique()

    print(f"Total Patients in Metadata: {total_valid_patients} | Total Samples: {total_valid_samples}")
    print(f"Train + Val: {len(df_train_val)} samples ({df_train_val['patient_id'].nunique()} patients - {len(df_train_val)/total_valid_samples:.1%})")
    print(f"Train: {len(df_train)} samples ({df_train['patient_id'].nunique()} patients - {len(df_train)/total_valid_samples:.1%})")
    print(f"Val:   {len(df_val)} samples ({df_val['patient_id'].nunique()} patients - {len(df_val)/total_valid_samples:.1%})")
    print(f"Test:  {len(df_test)} samples ({df_test['patient_id'].nunique()} patients - {len(df_test)/total_valid_samples:.1%})")

    print("\n" + "-"*50)
    print(f"TARGET ('{target_col_metadata}') DISTRIBUTION across splits (Sample-level %):")
    print("-"*50)
    target_dist = pd.DataFrame({
        'Train': df_train[target_col_metadata].value_counts(normalize=True),
        'Val': df_val[target_col_metadata].value_counts(normalize=True),
        'Test': df_test[target_col_metadata].value_counts(normalize=True)
    }).round(3)
    print(target_dist)


if __name__ == "__main__":
    print("Loading metadata and clinical data...")
    metadata_table = pd.read_csv(metadata_filepath, sep='\t')
    clinical_table = pd.read_csv(clinical_filepath, sep='\t')

    metadata_table_cleaned, clinical_table_cleaned = clean_feature_cols(metadata_table, clinical_table)
    metadata_table_cleaned, clinical_table_cleaned = clean_and_filter_target(metadata_table_cleaned, clinical_table_cleaned)

    perform_split_and_save(metadata_table_cleaned, clinical_table_cleaned)

    print("\nDONE!\n")
