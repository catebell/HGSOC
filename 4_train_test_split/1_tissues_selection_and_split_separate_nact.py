import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import config_file_tree

"""
Split based on metadata and patient_id + metadata and clinical processing
"""

os.makedirs(config_file_tree.SPLIT_DIR, exist_ok=True)
log_filepath = os.path.join(config_file_tree.SPLIT_DIR, 'filter_and_split_applied.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filepath, mode='w'),
        logging.StreamHandler()
    ]
)

""""" paths and parameters """""

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
graph_dir = config_file_tree.PROCESSED_SAMPLES_DIR

metadata_filepath = os.path.join(graph_dir, 'associated_metadata.tsv')
clinical_filepath = os.path.join(preprocessed_dir, "preprocessed_clinical_data.tsv")

output_metadata_filepath = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
output_clinical_filepath = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")

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

SUBSELECT_TISSUES = True       # Enable subselection filtering for sample/tissue-level metadata
SUBSELECT_PATIENTS = True      # Enable subselection filtering for patient-level task
PRIMARY_TREATMENT_STRATEGY = 'pds'   # Coorte principale su cui fare train/val/test
EXTRACT_SECONDARY_NACT = True        # Estrae e salva separatamente i pazienti NACT per Zero-Shot / Fine-tuning


CONDENSATE = getattr(config_file_tree, 'CONDENSATE', False)


def subselection_tissues(df_metadata):
    """ Exclude some samples based on metadata features. """
    if 'phase_label' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['phase_label'] == 'Primary'].copy()
        logging.info(f"\n[FILTER-METADATA] Removed {n_before - len(df_metadata)} samples with phase_label != 'Primary' (Interval, Relapse).")

    if 'sample_class' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['sample_class'] == 'BulkTissue'].copy()
        logging.info(f"\n[FILTER-METADATA] Removed {n_before - len(df_metadata)} samples with sample_class != 'BulkTissue' (CellLines, OrganoidLines, ...).")

    if 'treatment' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['treatment'] == 'Untreated'].copy()
        logging.info(f"\n[FILTER-METADATA] Removed {n_before - len(df_metadata)} samples with treatment != 'Untreated'.")

    return df_metadata


def subselection_patients(df_clinical, strategy='pds'):
    filter_col = 'Treatment_strategy'
    n_patients = len(df_clinical)
    df_clinical = df_clinical[df_clinical[filter_col] == strategy].copy()
    logging.info(
        f"\n[FILTER-CLINICAL] Removed {n_patients - len(df_clinical)} patients with {filter_col} != {strategy}.")

    # Applica il filtro NACT_cycles solo se si stanno selezionando pazienti NACT TODO serve o no
    '''if strategy == 'nact' and 'NACT_cycles' in df_clinical.columns:
        filter_col = 'NACT_cycles'
        filter_value = 2
        n_patients = len(df_clinical)
        df_clinical = df_clinical[df_clinical[filter_col] >= filter_value].copy()
        logging.info(
            f"\n[FILTER-CLINICAL] Removed {n_patients - len(df_clinical)} patients with {filter_col} < {filter_value}.")'''

    return df_clinical


def extract_and_save_nact_cohort(df_metadata_clean, df_clinical_raw):
    """
    Estrae i pazienti NACT e i relativi campioni, salvando:
    1) clinical_data_nact.tsv e samples_nact.csv (intera coorte NACT per valutazione Zero-Shot)
    2) samples_nact_train_val.csv e samples_nact_test.csv (stratificati su target_col_clinical per fine-tuning)
    """
    logging.info("\n" + "=" * 50)
    logging.info("EXTRACTING SEPARATE NACT COHORT FOR TRANSFER / EVAL")
    logging.info("=" * 50)

    df_nact = df_clinical_raw[df_clinical_raw['Treatment_strategy'] == 'nact'].copy()
    df_nact = df_nact.dropna(subset=[target_col_clinical]).copy()

    # Mantieni solo pazienti con almeno un campione nei metadati filtrati
    valid_patients = set(df_metadata_clean['patient_id'].astype(str).str.strip().str.upper().unique())
    df_nact = df_nact[df_nact['patient_id'].isin(valid_patients)].copy()

    cols_to_keep_clinical = ['patient_id']
    if target_col_clinical in df_nact.columns:
        cols_to_keep_clinical.append(target_col_clinical)
    if config_file_tree.USE_CLINICAL_DATA:
        cols_to_keep_clinical += [c for c in config_file_tree.CLINICAL_FEATURES if c not in cols_to_keep_clinical]

    df_nact_filtered = df_nact[cols_to_keep_clinical].copy()
    df_nact_samples = df_metadata_clean[df_metadata_clean['patient_id'].isin(df_nact_filtered['patient_id'])].copy()

    # Salva dataset completo NACT (per valutazione Zero-Shot su tutti i ~78 pazienti)
    df_nact_filtered.to_csv("clinical_data_nact.tsv", sep='\t', index=False)
    df_nact_samples.to_csv("samples_nact.csv", index=False)

    logging.info(f"NACT Cohort saved: {len(df_nact_filtered)} patients, {len(df_nact_samples)} samples.")
    logging.info(f"NACT '{target_col_clinical}' distribution:\n{df_nact_filtered[target_col_clinical].value_counts()}")

    # Crea anche uno split 85% Train+Val / 15% Test stratificato DIRETTAMENTE sul target clinico
    # (evita l'inversione delle classi causata dal fallback su tissue_type)
    nact_train_val_pts, nact_test_pts = train_test_split(
        df_nact_filtered,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df_nact_filtered[target_col_clinical]
    )

    df_nact_samples_tv = df_nact_samples[df_nact_samples['patient_id'].isin(nact_train_val_pts['patient_id'])].copy()
    df_nact_samples_test = df_nact_samples[df_nact_samples['patient_id'].isin(nact_test_pts['patient_id'])].copy()

    df_nact_samples_tv.to_csv("samples_nact_train_val.csv", index=False)
    df_nact_samples_test.to_csv("samples_nact_test.csv", index=False)

    logging.info(
        f"NACT Stratified Split -> Train+Val: {len(nact_train_val_pts)} patients ({len(df_nact_samples_tv)} samples) | "
        f"Test: {len(nact_test_pts)} patients ({len(df_nact_samples_test)} samples)"
    )
    logging.info("=" * 50 + "\n")


def clean_feature_cols(df_metadata, df_clinical):
    """ Keep only cols needed for sample metadata. """

    # SAMPLES FEATURES
    if SUBSELECT_TISSUES:
        df_metadata = subselection_tissues(df_metadata)

    cols_to_keep_metadata = ['sample_id', 'patient_id']
    if target_col_metadata in df_metadata.columns:
        cols_to_keep_metadata.append(target_col_metadata)

    if config_file_tree.USE_SAMPLES_METADATA:
        df_metadata_filtered = df_metadata[cols_to_keep_metadata + config_file_tree.METADATA_FEATURES].copy()
    else:
        df_metadata_filtered = df_metadata[cols_to_keep_metadata].copy()

    logging.info(f"Metadata cleaned, kept cols: {df_metadata_filtered.columns.to_list()}")

    # Estrai e salva i pazienti NACT prima di filtrare la coorte principale su PDS
    if EXTRACT_SECONDARY_NACT:
        extract_and_save_nact_cohort(df_metadata_filtered, df_clinical)

    # PATIENT FEATURES (Coorte principale: PDS)
    if SUBSELECT_PATIENTS:
        df_clinical = subselection_patients(df_clinical, strategy=PRIMARY_TREATMENT_STRATEGY)

    cols_to_keep_clinical = ['patient_id']
    if target_col_clinical in df_clinical.columns:
        cols_to_keep_clinical.append(target_col_clinical)

    if config_file_tree.USE_CLINICAL_DATA:
        df_clinical_filtered = df_clinical[cols_to_keep_clinical + config_file_tree.CLINICAL_FEATURES].copy()
    else:
        df_clinical_filtered = df_clinical[cols_to_keep_clinical].copy()

    logging.info(f"Clinical data cleaned, kept cols: {df_clinical_filtered.columns.to_list()}\n")

    return df_metadata_filtered, df_clinical_filtered


def clean_and_filter_target(df_metadata, df_clinical):
    """ Exclude target NaNs and rare classes, saving remaining samples. """

    logging.info(f"\nDeduplicated samples in mapping: {len(df_metadata)} ({df_metadata['patient_id'].nunique()} patients)")

    if CONDENSATE:
        logging.info(f"\n[CONDENSATE=True] Assigning patient target '{target_col_clinical}' to each sample metadata...")
        patient_target_map = df_clinical.set_index('patient_id')[target_col_clinical].to_dict()
        df_metadata[target_col_clinical] = df_metadata['patient_id'].map(patient_target_map)

    initial_len = len(df_metadata)
    df_clean = df_metadata.dropna(subset=[target_col_metadata]).copy()
    if len(df_clean) < initial_len:
        logging.info(f"[FILTER-METADATA] Dropped {initial_len - len(df_clean)} samples with NaN in target '{target_col_metadata}'.")

    """"" Removing rare classes (by patient or sample) """""
    all_classes = df_clean[target_col_metadata].value_counts()

    logging.info(f"\n(Patient-level) rare sample classes filtering...")
    df_patients = df_clean[['patient_id', target_col_metadata]].drop_duplicates(subset=['patient_id'], keep='first')
    patients_per_class = df_patients[target_col_metadata].value_counts()
    valid_classes = patients_per_class[patients_per_class >= MIN_PATIENTS_PER_CLASS].index
    removed_classes = set(all_classes.index) - set(valid_classes)

    if removed_classes:
        logging.info(f"[FILTER-METADATA] Removed sample classes with < {MIN_PATIENTS_PER_CLASS} unique patients: {removed_classes}\n")
        df_clean = df_clean[df_clean[target_col_metadata].isin(valid_classes)].copy()

    # CLINICAL
    n_before_drop = len(df_clinical)
    df_clinical = df_clinical.dropna(subset=[target_col_clinical]).copy()
    if len(df_clinical) < n_before_drop:
        logging.info(
            f"[FILTER-CLINICAL] Dropped {n_before_drop - len(df_clinical)} clinical records with NaN in patient target '{target_col_clinical}'.\n")

    clinical_class_counts = df_clinical[target_col_clinical].value_counts()
    valid_clinical_classes = clinical_class_counts[clinical_class_counts >= MIN_PATIENTS_PER_CLASS].index
    removed_clinical_classes = set(clinical_class_counts.index) - set(valid_clinical_classes)

    if removed_clinical_classes:
        logging.info(
            f"[FILTER-CLINICAL] Removed patient target classes with < {MIN_PATIENTS_PER_CLASS} patients: {removed_clinical_classes}\n")
        df_clinical = df_clinical[df_clinical[target_col_clinical].isin(valid_clinical_classes)].copy()

    valid_patients = set(df_clean['patient_id'].astype(str).str.strip().str.upper().unique())
    n_patients = len(df_clinical)
    df_clinical = df_clinical[df_clinical['patient_id'].isin(valid_patients)].copy()
    logging.info(f"\n[FILTER-CLINICAL] Removed {n_patients - len(df_clinical)} patients with no samples in processed metadata.")

    if task == 'survival':
        if time_col and time_col in df_clean.columns:
            n_before_time = len(df_clean)
            df_clean = df_clean[df_clean[time_col] > 0].dropna(subset=[time_col]).copy()
            logging.info(f"[FILTER] Removed {n_before_time - len(df_clean)} samples with invalid/missing time col '{time_col}'.")
        logging.info(f"\n--- Sample Task: SURVIVAL ---\n")
        logging.info(f"Event Col: '{target_col_metadata}' | Time Col: '{time_col}'")
        logging.info(f"Event distribution between samples:\n{df_clean[target_col_metadata].value_counts(normalize=True)}")

    elif task == 'classification':
        logging.info(f"\n--- Sample Task: CLASSIFICATION ---\n")
        logging.info(f"Sample Target Label: '{target_col_metadata}'")
        logging.info(f"(Sample-level) Classes distribution:\n{df_clean[target_col_metadata].value_counts(normalize=True)}")

    logging.info(f"\nPatient Target Label: '{target_col_clinical}'")
    logging.info(f"Classes distribution (Patient level):\n{df_clinical[target_col_clinical].value_counts(normalize=True)}")

    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    logging.info(f"\nSaved processed metadata ({df_clean.shape}) to {output_metadata_filepath}")

    df_clinical.to_csv(output_clinical_filepath, sep='\t', index=False)
    logging.info(f"Saved processed clinical data ({df_clinical.shape}) to {output_clinical_filepath}\n")

    return df_clean, df_clinical


def perform_split_and_save(df_clean, df_clinical):
    """ Split based on metadata, stratifying by patient or sample. """

    ratio_val_test = VAL_SIZE + TEST_SIZE
    ratio_test = TEST_SIZE / ratio_val_test

    logging.info(f"\n[PATIENT-LEVEL stratified split]")
    df_patients = df_clean[['patient_id', target_col_metadata]].drop_duplicates(subset=['patient_id'], keep='first')

    df_patients = df_patients.merge(
        df_clinical[['patient_id', target_col_clinical]],
        on='patient_id',
        how='left'
    )
    df_patients['composite_target'] = (
        df_patients[target_col_metadata].astype(str) + "_" +
        df_patients[target_col_clinical].fillna('unfiltered').astype(str)
    )

    stratification_target = 'composite_target'
    counts = df_patients[stratification_target].value_counts()

    if counts.min() < 2:
        logging.info("[WARNING] Rare composite target classes detected (<2 patients). Falling back to sample target stratification.")
        stratification_target = target_col_metadata
        counts = df_patients[stratification_target].value_counts()
        if counts.min() < 2:
            stratification_target = None

    train_pts_df, val_test_pts_df = train_test_split(
        df_patients,
        test_size=ratio_val_test,
        random_state=RANDOM_SEED,
        stratify=df_patients[stratification_target] if stratification_target else None
    )

    if stratification_target:
        vt_counts = val_test_pts_df[stratification_target].value_counts()
        strat_vt = val_test_pts_df[stratification_target] if vt_counts.min() >= 2 else None
    else:
        strat_vt = None

    val_pts_df, test_pts_df = train_test_split(
        val_test_pts_df,
        test_size=ratio_test,
        random_state=RANDOM_SEED,
        stratify=strat_vt
    )

    df_train = df_clean[df_clean['patient_id'].isin(train_pts_df['patient_id'])].copy()
    df_val = df_clean[df_clean['patient_id'].isin(val_pts_df['patient_id'])].copy()
    df_test = df_clean[df_clean['patient_id'].isin(test_pts_df['patient_id'])].copy()

    df_train_val = pd.DataFrame(pd.concat([df_train, df_val]))

    df_train_val.to_csv("samples_train_val.csv", index=False)
    df_train.to_csv("samples_train.csv", index=False)
    df_val.to_csv("samples_val.csv", index=False)
    df_test.to_csv("samples_test.csv", index=False)

    total_valid_samples = len(df_clean)
    total_valid_patients = df_clean['patient_id'].nunique()

    logging.info("\nSplit Summary (GAT Samples Metadata):")
    logging.info(f"Total Patients in Metadata: {total_valid_patients} | Total Samples: {total_valid_samples}")
    logging.info(f"Train + Val: {len(df_train_val)} samples ({df_train_val['patient_id'].nunique()} patients - {len(df_train_val)/total_valid_samples:.1%})")
    logging.info(f"Train: {len(df_train)} samples ({df_train['patient_id'].nunique()} patients - {len(df_train)/total_valid_samples:.1%})")
    logging.info(f"Val:   {len(df_val)} samples ({df_val['patient_id'].nunique()} patients - {len(df_val)/total_valid_samples:.1%})")
    logging.info(f"Test:  {len(df_test)} samples ({df_test['patient_id'].nunique()} patients - {len(df_test)/total_valid_samples:.1%})")

    logging.info("\n" + "-"*50)
    logging.info(f"TARGET ('{target_col_metadata}') DISTRIBUTION across splits (Sample-level %):")
    logging.info("-"*50)
    target_dist = pd.DataFrame({
        'Train': df_train[target_col_metadata].value_counts(normalize=True),
        'Val': df_val[target_col_metadata].value_counts(normalize=True),
        'Test': df_test[target_col_metadata].value_counts(normalize=True)
    }).round(3)
    logging.info(f"\n{target_dist}")

    if target_col_clinical in df_clinical.columns:
        df_clinical_train = df_clinical[df_clinical['patient_id'].isin(train_pts_df['patient_id'])]
        df_clinical_val = df_clinical[df_clinical['patient_id'].isin(val_pts_df['patient_id'])]
        df_clinical_test = df_clinical[df_clinical['patient_id'].isin(test_pts_df['patient_id'])]
        df_clinical_train_val = pd.concat([df_clinical_train, df_clinical_val])

        total_clin_patients = len(df_clinical)

        logging.info("\n" + "="*50)
        logging.info("Split Summary (MIL Patient Clinical Data):")
        logging.info("="*50)
        logging.info(f"Total Filtered Clinical Patients: {total_clin_patients}")
        logging.info(f"Train + Val: {len(df_clinical_train_val)} patients ({len(df_clinical_train_val)/total_clin_patients:.1%})")
        logging.info(f"Train: {len(df_clinical_train)} patients ({len(df_clinical_train)/total_clin_patients:.1%})")
        logging.info(f"Val:   {len(df_clinical_val)} patients ({len(df_clinical_val)/total_clin_patients:.1%})")
        logging.info(f"Test:  {len(df_clinical_test)} patients ({len(df_clinical_test)/total_clin_patients:.1%})")

        logging.info("\n" + "-"*50)
        logging.info(f"PATIENT TARGET ('{target_col_clinical}') DISTRIBUTION across splits (Patient-level %):")
        logging.info("-"*50)
        patient_target_dist = pd.DataFrame({
            'Train': df_clinical_train[target_col_clinical].value_counts(normalize=True),
            'Val': df_clinical_val[target_col_clinical].value_counts(normalize=True),
            'Test': df_clinical_test[target_col_clinical].value_counts(normalize=True)
        }).round(3)
        logging.info(f"\n{patient_target_dist}")


if __name__ == "__main__":
    logging.info("Loading metadata and clinical data...")
    metadata_table = pd.read_csv(metadata_filepath, sep='\t')
    clinical_table = pd.read_csv(clinical_filepath, sep='\t')

    metadata_table_cleaned, clinical_table_cleaned = clean_feature_cols(metadata_table, clinical_table)
    metadata_table_cleaned, clinical_table_cleaned = clean_and_filter_target(metadata_table_cleaned, clinical_table_cleaned)

    perform_split_and_save(metadata_table_cleaned, clinical_table_cleaned)

    logging.info("\nDONE!\n")