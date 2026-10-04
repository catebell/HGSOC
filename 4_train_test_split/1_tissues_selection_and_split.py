import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

import config_file_tree

"""
Split based on metadata
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

""""" paths and parameters """""

split_by_patient = True  # multiple tissues belonging to the same patient must end up in the same split

graph_dir = config_file_tree.PROCESSED_GRAPHS_DIR
metadata_filepath = os.path.join(graph_dir, 'associated_metadata.tsv')

output_metadata_filepath = "metadata_for_training.tsv"

TASK = config_file_tree.TASK
TARGET_COL = config_file_tree.TARGET_COL  # task label to stratify on
TIME_COL = ""

if TASK == 'survival':
    TIME_COL = config_file_tree.TIME_COL


# 70% Train, 15% Val, 15% Test
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 7

MIN_PATIENTS_PER_CLASS = 5


""""" samples selection and deduplication """""

df_metadata = pd.read_csv(metadata_filepath, sep="\t")

# samples filtering by extraction and priority for duplicates
df_metadata['ext_priority'] = df_metadata['extraction'].str.extract(r'(\d+)').fillna(99).astype(int)
df_metadata = df_metadata.sort_values(by=['patient_id', 'phase', 'tissue', 'ext_priority'])
df_metadata = df_metadata.drop_duplicates(subset=['patient_id', 'phase', 'tissue'], keep='first')
df_metadata = df_metadata.drop(columns=['ext_priority'])

print(f"Deduplicated samples in mapping: {len(df_metadata)} ({df_metadata['patient_id'].nunique()} patients)")
print("-" * 50)


df_clean = df_metadata

if TASK == 'survival':
    print(f"\n--- Task: SURVIVAL ---\n")
    print(f"Event Col: '{TARGET_COL}' | Time Col: '{TIME_COL}'")
    print("Event distribution between samples:\n", df_clean[TARGET_COL].value_counts(normalize=True))

elif TASK == 'classification':
    print(f"\n--- Task: CLASSIFICATION ---\n")
    print(f"Target Label: '{TARGET_COL}'")
    print("Classes distribution (Sample level):\n", df_clean[TARGET_COL].value_counts(normalize=True))


""""" stratified split """""

ratio_val_test = VAL_SIZE + TEST_SIZE  # (0.15 + 0.15 = 0.30 --> 30%)
ratio_test = TEST_SIZE / ratio_val_test  # 0.50

if split_by_patient:
    print(f"\n[PATIENT-LEVEL stratified split]")
    all_classes = df_clean[TARGET_COL].value_counts()

    df_patients = df_clean[['patient_id', TARGET_COL]].drop_duplicates(subset=['patient_id'], keep='first')

    patients_per_class = df_patients[TARGET_COL].value_counts()
    valid_classes = patients_per_class[patients_per_class >= MIN_PATIENTS_PER_CLASS].index
    removed_classes = set(all_classes.index) - set(valid_classes)

    if removed_classes:
        print(f"[FILTER] Removed classes with <  {MIN_PATIENTS_PER_CLASS} unique patients: {removed_classes}")
        df_patients = df_patients[df_patients[TARGET_COL].isin(valid_classes)].copy()
        df_clean = df_metadata[df_metadata[TARGET_COL].isin(valid_classes)].copy()

    # split train / val+test
    train_pts_df, val_test_pts_df = train_test_split(
        df_patients,
        test_size=ratio_val_test,
        random_state=RANDOM_SEED,
        stratify=df_patients[TARGET_COL]
    )

    # split val / test (remaining patients)
    val_pts_df, test_pts_df = train_test_split(
        val_test_pts_df,
        test_size=ratio_test,
        random_state=RANDOM_SEED,
        stratify=val_test_pts_df[TARGET_COL]
    )

    # patient mapping to retrieve related tissues
    df_train = df_clean[df_clean['patient_id'].isin(train_pts_df['patient_id'])].copy()
    df_val   = df_clean[df_clean['patient_id'].isin(val_pts_df['patient_id'])].copy()
    df_test  = df_clean[df_clean['patient_id'].isin(test_pts_df['patient_id'])].copy()
else:
    print(f"\n[SAMPLE-LEVEL stratified split]")

    class_counts = df_metadata[TARGET_COL].value_counts()
    valid_classes = class_counts[class_counts >= MIN_PATIENTS_PER_CLASS].index
    removed_classes = set(class_counts.index) - set(valid_classes)

    if removed_classes:
        print(f"[FILTER] Removed classes with < {MIN_PATIENTS_PER_CLASS} samples: {removed_classes}")
        df_clean = df_metadata[df_metadata[TARGET_COL].isin(valid_classes)].copy()

    # split train / val+test (sample level)
    df_train, df_val_test = train_test_split(
        df_clean,
        test_size=ratio_val_test,
        random_state=RANDOM_SEED,
        stratify=df_clean[TARGET_COL]
    )

    # split val / test (sample level)
    df_val, df_test = train_test_split(
        df_val_test,
        test_size=ratio_test,
        random_state=RANDOM_SEED,
        stratify=df_val_test[TARGET_COL]
    )

df_train_val = pd.DataFrame(pd.concat([df_train, df_val]))

df_train_val.to_csv("samples_train_val.csv", index=False)
df_train.to_csv("samples_train.csv", index=False)
df_val.to_csv("samples_val.csv", index=False)
df_test.to_csv("samples_test.csv", index=False)
df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)

print("\nSplit:")

total_valid_samples = len(df_clean)
total_valid_patients = df_clean['patient_id'].nunique()

print(f"Total Patients: {total_valid_patients} | Total Samples: {total_valid_samples}")
print(f"Train + Val: {len(df_train_val)} samples ({df_train_val['patient_id'].nunique()} patients - {len(df_train_val)/total_valid_samples:.1%})")
print(f"    Train: {len(df_train)} samples ({df_train['patient_id'].nunique()} patients - {len(df_train)/total_valid_samples:.1%})")
print(f"    Val:   {len(df_val)} samples ({df_val['patient_id'].nunique()} patients - {len(df_val)/total_valid_samples:.1%})")
print(f"Test:  {len(df_test)} samples ({df_test['patient_id'].nunique()} patients - {len(df_test)/total_valid_samples:.1%})")


print("\n" + "-"*50)
print(f"TARGET ('{TARGET_COL}') DISTRIBUTION across splits (Sample-level %):")
print("-"*50)
target_dist = pd.DataFrame({
    'Train': df_train[TARGET_COL].value_counts(normalize=True),
    'Val': df_val[TARGET_COL].value_counts(normalize=True),
    'Test': df_test[TARGET_COL].value_counts(normalize=True)
}).round(3)
print(target_dist)
