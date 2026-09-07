import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config_file_tree

'''
Split at patient level: multiple tissues belonging to the same patient must end up in the same split.
'''


""""" paths and parameters """""

dataset_dir_path = config_file_tree.DATASET_DIR
metadata_filepath = os.path.join(dataset_dir_path, 'samples_metadata.tsv')

TASK = config_file_tree.TASK
TARGET_COL, TIME_COL = "", ""  # task label to stratify on
cols_to_fetch = []

if TASK == 'classification':
    TARGET_COL = config_file_tree.TARGET_COL
    cols_to_fetch = [TARGET_COL]

elif TASK == 'survival':
    TARGET_COL = config_file_tree.SURVIVAL_EVENT_COL
    TIME_COL = config_file_tree.SURVIVAL_TIME_COL
    cols_to_fetch = [TARGET_COL, TIME_COL]


# 70% Train, 15% Val, 15% Test
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 7


""""" samples selection by cohort """""

df_metadata = pd.read_csv(metadata_filepath, sep="\t")
df_metadata = df_metadata[df_metadata['cohort'] == 'H'].copy()

# samples filtering by extraction and priority for duplicates
df_metadata['ext_priority'] = df_metadata['extraction'].str.extract(r'(\d+)').fillna(99).astype(int)
df_metadata = df_metadata.sort_values(by=['patient_id', 'phase', 'tissue', 'ext_priority'])
df_metadata = df_metadata.drop_duplicates(subset=['patient_id', 'phase', 'tissue'], keep='first')
df_metadata = df_metadata.drop(columns=['ext_priority'])


""""" tissues/phases single/multiple separation """""

patient_phase_counts = df_metadata.groupby('patient_id')['phase'].nunique()
multiple_phase_patients = patient_phase_counts[patient_phase_counts > 1].index  # patients with more clinical phases
df_test_multiple_phases = df_metadata[df_metadata['patient_id'].isin(multiple_phase_patients)].copy()

# single samples (for Train/Val/Test)  # TODO check num_samples e cambiare questa logica: train su tutto o in base a fase? Più livelli?
df_single_tissue_set = df_metadata[~df_metadata['patient_id'].isin(multiple_phase_patients)].copy()

df_single_tissue_set.to_csv('tissues_single_phase.csv', index=False)
df_test_multiple_phases.to_csv('tissues_multiple_phases.csv', index=False)

print(f"Cohort H - Single phase samples: {len(df_single_tissue_set)} samples ({df_single_tissue_set['patient_id'].nunique()} patients)")
print(f"Cohort H - Multiple phase samples:  {len(df_test_multiple_phases)} samples ({len(multiple_phase_patients)} patients)\n")


""""" single tissues train/test separation """""

# drop nan in target col (not useful)
for col in cols_to_fetch:
    df_single_tissue_set[col] = df_single_tissue_set[col].replace(-1, np.nan)

df_singles_clean = df_single_tissue_set.dropna(subset=cols_to_fetch).copy()

if TASK == 'survival':
    print(f"--- Task: SURVIVAL ---")
    print(f"Event Col: '{TARGET_COL}' | Time Col: '{TIME_COL}'")
    df_singles_clean = df_singles_clean[df_singles_clean[TIME_COL] > 0]  # remove non-valid times
    df_singles_clean[TARGET_COL] = df_singles_clean[TARGET_COL].astype(int)

    print("Event status distribution:\n", df_singles_clean[TARGET_COL].value_counts(normalize=True))

elif TASK == 'classification':
    print(f"--- Task: CLASSIFICATION ---")
    print(f"Target Label: '{TARGET_COL}'")

    #TODO in clinical preprocessing
    df_singles_clean[TARGET_COL] = df_singles_clean[TARGET_COL].str.replace(r'\d+$', '', regex=True)

    min_patients_per_class = 7
    # drop cases with less than min_patients_per_class occurrences (not stratifiable)
    patient_counts_per_class = df_singles_clean.groupby(TARGET_COL)['patient_id'].nunique()
    valid_classes = patient_counts_per_class[patient_counts_per_class >= min_patients_per_class].index
    df_singles_clean = df_singles_clean[df_singles_clean[TARGET_COL].isin(valid_classes)].copy()

    print("Classes distribution (Sample level):\n", df_singles_clean[TARGET_COL].value_counts(normalize=True))

print(f"Initial samples in mapping: {len(df_single_tissue_set)}")
print(f"Valid samples with not-null label: {len(df_singles_clean)}")
print("-" * 40)


""""" stratified split (patient level) """""

# less frequent classes with more priority (TODO check)
class_counts = df_singles_clean[TARGET_COL].value_counts()
df_singles_clean['class_freq'] = df_singles_clean[TARGET_COL].map(class_counts)
df_sorted = df_singles_clean.sort_values(by='class_freq', ascending=True)

# patient table with target class
df_patients = df_sorted[['patient_id', TARGET_COL]].drop_duplicates(subset=['patient_id'], keep='first')
df_singles_clean = df_singles_clean.drop(columns=['class_freq'])

# split train / val+test --> 0.15 + 0.15 = 0.30 (30%)
ratio_val_test = VAL_SIZE + TEST_SIZE  # 0.30
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
df_train = df_singles_clean[df_singles_clean['patient_id'].isin(train_pts)]
df_val   = df_singles_clean[df_singles_clean['patient_id'].isin(val_pts)]
df_test  = df_singles_clean[df_singles_clean['patient_id'].isin(test_pts)]

df_train_val = pd.concat([df_train, df_val])

df_train_val.to_csv("singles_train_val.csv", index=False)
df_train.to_csv("singles_train.csv", index=False)
df_val.to_csv("singles_val.csv", index=False)
df_test.to_csv("singles_test.csv", index=False)


print("\nSplit:")

total_valid_samples = len(df_singles_clean)
total_valid_pts = df_singles_clean['patient_id'].nunique()

print(f"Total Patients: {total_valid_pts} | Total Samples: {total_valid_samples}")
print(f"Train + Val: {len(df_train_val)} samples ({df_train_val['patient_id'].nunique()} patients - {len(df_train_val)/total_valid_samples:.1%})")
print(f"    Train: {len(df_train)} samples ({df_train['patient_id'].nunique()} patients - {len(df_train)/total_valid_samples:.1%})")
print(f"    Val:   {len(df_val)} samples ({df_val['patient_id'].nunique()} patients - {len(df_val)/total_valid_samples:.1%})")
print(f"Test:  {len(df_test)} samples ({df_test['patient_id'].nunique()} patients - {len(df_test)/total_valid_samples:.1%})")
