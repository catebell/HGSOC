import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

import config_file_tree

"""
Split based on metadata
"""

""""" paths and parameters """""

clinical_cols_to_extract = config_file_tree.CLINICAL_FEATURES

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
graph_dir = config_file_tree.PROCESSED_GRAPHS_DIR

metadata_filepath = os.path.join(graph_dir, 'associated_metadata.tsv')
clinical_filepath = os.path.join(preprocessed_dir, "preprocessed_clinical_data.tsv")

output_metadata_filepath = "metadata_for_training.tsv"


target_col = config_file_tree.TARGET_COL  # task label to stratify on
task = config_file_tree.TASK

time_col = config_file_tree.TIME_COL if task == 'survival' else ""

split_by_patient = True  # multiple tissues belonging to the same patient must end up in the same split

# 70% Train, 15% Val, 15% Test
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 7

MIN_PATIENTS_PER_CLASS = 5


def filter_metadata(df_metadata):
    """ Keep only cols needed. """
    cols_to_keep = ['sample_id', 'patient_id']

    if target_col in df_metadata.columns:
        cols_to_keep.append(target_col)

    if config_file_tree.USE_SAMPLES_METADATA:
        df_filtered = df_metadata[cols_to_keep + config_file_tree.METADATA_FEATURES].copy()
    else:
        df_filtered = df_metadata[cols_to_keep].copy()

    print(f"Filtering metadata, kept cols: {df_filtered.columns.to_list()}\n")
    return df_filtered


def enrich_metadata_with_clinical_features(df_metadata):
    """ Add patients features to metadata if needed. """
    df_clinical = pd.read_csv(clinical_filepath, sep='\t')

    patient_key_clinical = df_clinical.columns[0]
    cols_to_keep = [patient_key_clinical]

    if config_file_tree.USE_CLINICAL_DATA:
        cols_to_keep += list(clinical_cols_to_extract)

    if target_col in df_clinical.columns and target_col not in cols_to_keep:
        cols_to_keep.append(target_col)

    if target_col in cols_to_keep:
        print(f"Target feature from clinical data: {target_col}\n")
    else:
        if target_col not in df_metadata.columns:
            raise KeyError("Target column neither in metadata nor in clinical data.")

    existing_cols = [col for col in cols_to_keep if col in df_clinical.columns]
    missing_cols = set(cols_to_keep) - set(existing_cols)
    if missing_cols:
        print(f"[WARNING] The following specified columns were not found in clinical file: {missing_cols}")

    cols_for_merge = list(dict.fromkeys(existing_cols))
    df_clinical_sub = df_clinical[cols_for_merge].copy()

    # avoid cols duplication (es. _x, _y) if re-execution
    for col in existing_cols:
        if col in df_metadata.columns:
            df_metadata.drop(columns=[col], inplace=True)

    df_clinical_sub[patient_key_clinical] = df_clinical_sub[patient_key_clinical].str.upper()
    df_metadata['patient_id'] = df_metadata['patient_id'].str.upper()

    print(f"Merging clinical data on: 'patient_id' (metadata) <-> '{patient_key_clinical}' (clinical)...")
    df_merged = df_metadata.merge(
        df_clinical_sub,
        left_on='patient_id',
        right_on=patient_key_clinical,
        how='left'
    )

    # check if some samples remain unmatched
    na_samples = df_merged[df_merged[patient_key_clinical].isna()]
    if len(na_samples) > 0:
        df_merged = df_merged.dropna(subset=[patient_key_clinical])
        print(f"Excluded {len(na_samples)} samples because no clinical match found.")

    # redundant patient ID col exclusion
    if patient_key_clinical != 'patient_id' and patient_key_clinical in df_merged.columns:
        df_merged.drop(columns=[patient_key_clinical], inplace=True)

    print(f"\nClinical and Metadata merged successfully!")
    print(f"    New dataframe shape: {df_merged.shape}")
    print(f"    Clinical features added: {existing_cols}")

    return df_merged


def clean_and_filter_target(df_metadata):
    """ Exclude target NaNs and rare classes, saving remainning samples. """
    print(f"\nDeduplicated samples in mapping: {len(df_metadata)} ({df_metadata['patient_id'].nunique()} patients)")
    print("-" * 50)

    initial_len = len(df_metadata)
    df_clean = df_metadata.dropna(subset=[target_col]).copy()
    if len(df_clean) < initial_len:
        print(f"[FILTER] Dropped {initial_len - len(df_clean)} samples with NaN in target '{target_col}'.")

    if task == 'survival':
        if time_col and time_col in df_clean.columns:
            n_before_time = len(df_clean)
            df_clean = df_clean[df_clean[time_col] > 0].dropna(subset=[time_col]).copy()
            print(f"[FILTER] Removed {n_before_time - len(df_clean)} samples with invalid/missing time col '{time_col}'.")
        print(f"\n--- Task: SURVIVAL ---\n")
        print(f"Event Col: '{target_col}' | Time Col: '{time_col}'")
        print("Event distribution between samples:\n", df_clean[target_col].value_counts(normalize=True))

    elif task == 'classification':
        print(f"\n--- Task: CLASSIFICATION ---\n")
        print(f"Target Label: '{target_col}'")
        print("Classes distribution (Sample level):\n", df_clean[target_col].value_counts(normalize=True))


    """"" Removing rare classes (by patient or sample) """""

    all_classes = df_clean[target_col].value_counts()

    if split_by_patient:
        print(f"\n[PATIENT-LEVEL rare classes filtering]")
        df_patients = df_clean[['patient_id', target_col]].drop_duplicates(subset=['patient_id'], keep='first')
        patients_per_class = df_patients[target_col].value_counts()
        valid_classes = patients_per_class[patients_per_class >= MIN_PATIENTS_PER_CLASS].index
        removed_classes = set(all_classes.index) - set(valid_classes)

        if removed_classes:
            print(f"[FILTER] Removed classes with < {MIN_PATIENTS_PER_CLASS} unique patients: {removed_classes}")
            df_clean = df_clean[df_clean[target_col].isin(valid_classes)].copy()
    else:
        print(f"\n[SAMPLE-LEVEL rare classes filtering]")
        class_counts = df_clean[target_col].value_counts()
        valid_classes = class_counts[class_counts >= MIN_PATIENTS_PER_CLASS].index
        removed_classes = set(class_counts.index) - set(valid_classes)

        if removed_classes:
            print(f"[FILTER] Removed classes with < {MIN_PATIENTS_PER_CLASS} samples: {removed_classes}")
            df_clean = df_clean[df_clean[target_col].isin(valid_classes)].copy()

    # saving to file
    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    print(f"Saved filtered metadata to {output_metadata_filepath}")

    return df_clean


def perform_split_and_save(df_clean):
    """ Split stratifying by patient or sample. """

    ratio_val_test = VAL_SIZE + TEST_SIZE  # (0.15 + 0.15 = 0.30 --> 30%)
    ratio_test = TEST_SIZE / ratio_val_test  # 0.50

    if split_by_patient:
        print(f"\n[PATIENT-LEVEL stratified split]")
        df_patients = df_clean[['patient_id', target_col]].drop_duplicates(subset=['patient_id'], keep='first')

        # split train / val+test
        train_pts_df, val_test_pts_df = train_test_split(
            df_patients,
            test_size=ratio_val_test,
            random_state=RANDOM_SEED,
            stratify=df_patients[target_col]
        )

        # split val / test (remaining patients)
        val_pts_df, test_pts_df = train_test_split(
            val_test_pts_df,
            test_size=ratio_test,
            random_state=RANDOM_SEED,
            stratify=val_test_pts_df[target_col]
        )

        # patient mapping to retrieve related tissues
        df_train = df_clean[df_clean['patient_id'].isin(train_pts_df['patient_id'])].copy()
        df_val   = df_clean[df_clean['patient_id'].isin(val_pts_df['patient_id'])].copy()
        df_test  = df_clean[df_clean['patient_id'].isin(test_pts_df['patient_id'])].copy()
    else:
        print(f"\n[SAMPLE-LEVEL stratified split]")

        # split train / val+test (sample level)
        df_train, df_val_test = train_test_split(
            df_clean,
            test_size=ratio_val_test,
            random_state=RANDOM_SEED,
            stratify=df_clean[target_col]
        )

        # split val / test (sample level)
        df_val, df_test = train_test_split(
            df_val_test,
            test_size=ratio_test,
            random_state=RANDOM_SEED,
            stratify=df_val_test[target_col]
        )

    df_train_val = pd.DataFrame(pd.concat([df_train, df_val]))

    df_train_val.to_csv("samples_train_val.csv", index=False)
    df_train.to_csv("samples_train.csv", index=False)
    df_val.to_csv("samples_val.csv", index=False)
    df_test.to_csv("samples_test.csv", index=False)

    print("Split Summary:")
    total_valid_samples = len(df_clean)
    total_valid_patients = df_clean['patient_id'].nunique()

    print(f"Total Patients: {total_valid_patients} | Total Samples: {total_valid_samples}")
    print(f"Train + Val: {len(df_train_val)} samples ({df_train_val['patient_id'].nunique()} patients - {len(df_train_val)/total_valid_samples:.1%})")
    print(f"Train: {len(df_train)} samples ({df_train['patient_id'].nunique()} patients - {len(df_train)/total_valid_samples:.1%})")
    print(f"Val:   {len(df_val)} samples ({df_val['patient_id'].nunique()} patients - {len(df_val)/total_valid_samples:.1%})")
    print(f"Test:  {len(df_test)} samples ({df_test['patient_id'].nunique()} patients - {len(df_test)/total_valid_samples:.1%})")

    print("\n" + "-"*50)
    print(f"TARGET ('{target_col}') DISTRIBUTION across splits (Sample-level %):")
    print("-"*50)
    target_dist = pd.DataFrame({
        'Train': df_train[target_col].value_counts(normalize=True),
        'Val': df_val[target_col].value_counts(normalize=True),
        'Test': df_test[target_col].value_counts(normalize=True)
    }).round(3)
    print(target_dist)


if __name__ == "__main__":
    print("Loading metadata...")
    metadata_table = pd.read_csv(metadata_filepath, sep='\t')

    metadata_table = filter_metadata(metadata_table)
    metadata_table = enrich_metadata_with_clinical_features(metadata_table)
    metadata_table = clean_and_filter_target(metadata_table)
    perform_split_and_save(metadata_table)

    print("\nDONE!\n")