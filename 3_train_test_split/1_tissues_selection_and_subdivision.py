import os.path

import numpy as np
import pandas as pd
from narwhals import DataFrame
from sklearn.model_selection import train_test_split

import config_file_tree

'''
Split at patient level: multiple tissues belonging to the same patient must end up in the same split.
'''


""""" paths and parameters """""

preprocessed_data_dir = config_file_tree.PREPROCESSED_DATA_DIR
metadata_filepath = os.path.join(preprocessed_data_dir, 'preprocessed_samples_metadata.tsv')

TASK = config_file_tree.TASK
TARGET_COL, TIME_COL = "", ""  # task label to stratify on

if TASK == 'classification':
    TARGET_COL = config_file_tree.TARGET_COL

elif TASK == 'survival':
    TARGET_COL = config_file_tree.SURVIVAL_EVENT_COL
    TIME_COL = config_file_tree.SURVIVAL_TIME_COL


# 70% Train, 15% Val, 15% Test
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 7


""""" samples selection and deduplication """""

df_metadata = pd.read_csv(metadata_filepath, sep="\t")

# samples filtering by extraction and priority for duplicates
df_metadata['ext_priority'] = df_metadata['extraction'].str.extract(r'(\d+)').fillna(99).astype(int)
df_metadata = df_metadata.sort_values(by=['patient_id', 'phase', 'tissue', 'ext_priority'])
df_metadata = df_metadata.drop_duplicates(subset=['patient_id', 'phase', 'tissue'], keep='first')
df_metadata = df_metadata.drop(columns=['ext_priority'])

print(f"Deduplicated samples in mapping: {len(df_metadata)} ({df_metadata['patient_id'].nunique()} patients)")
print("-" * 50)


""""" tissues/phases single/multiple separation """""
'''
patient_phase_counts = df_metadata.groupby('patient_id')['phase'].nunique()
multiple_phase_patients = patient_phase_counts[patient_phase_counts > 1].index  # patients with more clinical phases
df_test_multiple_phases = df_metadata[df_metadata['patient_id'].isin(multiple_phase_patients)].copy()

# single samples (for Train/Val/Test)
df_single_tissue_set = df_metadata[~df_metadata['patient_id'].isin(multiple_phase_patients)].copy()

df_single_tissue_set.to_csv('tissues_single_phase.csv', index=False)
df_test_multiple_phases.to_csv('tissues_multiple_phases.csv', index=False)

print(f"Cohort H - Single phase samples: {len(df_single_tissue_set)} samples ({df_single_tissue_set['patient_id'].nunique()} patients)")
print(f"Cohort H - Multiple phase samples:  {len(df_test_multiple_phases)} samples ({len(multiple_phase_patients)} patients)\n")
'''

""""" single tissues train/test separation """""

df_clean = df_metadata.copy()

if TASK == 'survival':
    print(f"--- Task: SURVIVAL ---")
    print(f"Event Col: '{TARGET_COL}' | Time Col: '{TIME_COL}'")
    print("Event distribution between samples:\n", df_clean[TARGET_COL].value_counts(normalize=True))

elif TASK == 'classification':
    print(f"--- Task: CLASSIFICATION ---")
    print(f"Target Label: '{TARGET_COL}'")
    print("Classes distribution (Sample level):\n", df_clean[TARGET_COL].value_counts(normalize=True))


""""" stratified split (patient level) """""

df_patients = df_clean[['patient_id', TARGET_COL]].drop_duplicates(subset=['patient_id'], keep='first')

# split train / val+test
ratio_val_test = VAL_SIZE + TEST_SIZE  # (0.15 + 0.15 = 0.30 --> 30%)
train_pts, val_test_pts = train_test_split(
    df_patients['patient_id'],
    test_size=ratio_val_test,
    random_state=RANDOM_SEED,
    stratify=df_patients[TARGET_COL]
)

# split val / test (remaining patients)
val_test_targets = df_patients[df_patients['patient_id'].isin(val_test_pts)][TARGET_COL]
ratio_test = TEST_SIZE / ratio_val_test  # 0.50
val_pts, test_pts = train_test_split(
    val_test_pts,
    test_size=ratio_test,
    random_state=RANDOM_SEED,
    stratify=val_test_targets
)

# patient mapping to retrieve related tissues
df_train = df_clean[df_clean['patient_id'].isin(train_pts)].copy()
df_val   = df_clean[df_clean['patient_id'].isin(val_pts)].copy()
df_test  = df_clean[df_clean['patient_id'].isin(test_pts)].copy()

df_train_val = pd.DataFrame(pd.concat([df_train, df_val]))

df_train_val.to_csv("samples_train_val.csv", index=False)
df_train.to_csv("samples_train.csv", index=False)
df_val.to_csv("samples_val.csv", index=False)
df_test.to_csv("samples_test.csv", index=False)


print("\nSplit:")

total_valid_samples = len(df_clean)
total_valid_patients = df_clean['patient_id'].nunique()

print(f"Total Patients: {total_valid_patients} | Total Samples: {total_valid_samples}")
print(f"Train + Val: {len(df_train_val)} samples ({df_train_val['patient_id'].nunique()} patients - {len(df_train_val)/total_valid_samples:.1%})")
print(f"    Train: {len(df_train)} samples ({df_train['patient_id'].nunique()} patients - {len(df_train)/total_valid_samples:.1%})")
print(f"    Val:   {len(df_val)} samples ({df_val['patient_id'].nunique()} patients - {len(df_val)/total_valid_samples:.1%})")
print(f"Test:  {len(df_test)} samples ({df_test['patient_id'].nunique()} patients - {len(df_test)/total_valid_samples:.1%})")


print("\n" + "-"*50)
print("TISSUE TYPE DISTRIBUTION (Fluid vs Solid) across splits:")
print("-"*50)
for name, df_split in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    fluid_pct = df_split['is_fluid'].mean() * 100
    print(f"  {name:5s} -> Fluid (Asc/Plf): {fluid_pct:.1f}% | Solid: {100-fluid_pct:.1f}%")

print("\n" + "-"*50)
print(f"TARGET ('{TARGET_COL}') DISTRIBUTION across splits (Sample-level %):")
print("-"*50)
target_dist = pd.DataFrame({
    'Train': df_train[TARGET_COL].value_counts(normalize=True),
    'Val': df_val[TARGET_COL].value_counts(normalize=True),
    'Test': df_test[TARGET_COL].value_counts(normalize=True)
}).round(3)
print(target_dist)