from os import path
from sklearn.preprocessing import LabelEncoder
import config_file_tree
from TODO_utils_clinical_clean_methods import *

path_to_clinical_file = path.join('..', '1_dataset', 'data_HERCULES_DECIDER', 'samples_metadata.tsv')
df_clinical = pd.read_csv(path_to_clinical_file, sep="\t")


print(f"Clinical file loaded: {df_clinical.shape[0]} samples.")

# keeping only BulkTissue, filtering out CellLine and DissociatedTissue
if 'sample_class' in df_clinical.columns:
    df_clinical = df_clinical[df_clinical['sample_class'] == 'BulkTissue'].copy()
    print(f"'BulkTissue' samples kept: {df_clinical.shape[0]} samples.")

features_to_keep = [
    "sample_id",
    "patient_id",
    "cohort",
    "phase",
    "phase_label",
    "tissue",
    "extraction"
]

available_features = [col for col in features_to_keep if col in df_clinical.columns]
df_filtered = df_clinical[available_features].copy()


def simplify_tissue(tissue_str):
    """
    Groups tissues names removing numbers and L/R specifiers (L/R) #TODO check if keep
    (es. OvaR1 -> Ova, Ome1 -> Ome, Asc1 -> Asc).
    """
    if pd.isna(tissue_str):
        return 'Unknown'

    tissue_str = str(tissue_str).strip()

    # mapping known based on documentation
    main_tissues = ['Ome', 'Ova', 'Per', 'Asc', 'Tub', 'Ute', 'LN', 'Bow', 'Plf', 'Mes', 'Adn', 'Oth']
    for t in main_tissues:
        if tissue_str.startswith(t):
            return t
    return 'Other'


def clean_and_aggregate_features(df):
    df_clean = df.copy()

    if 'phase_label' in df_clean.columns:
        df_clean['phase_label'] = df_clean['phase_label'].fillna('Unknown')

    if 'tissue' in df_clean.columns:
        df_clean['tissue_group'] = df_clean['tissue'].apply(simplify_tissue)

    return df_clean


def encode_categorical_features(df):
    df_encoded = df.copy()

    cols_for_one_hot = ['cohort', 'phase_label', 'tissue_group']
    existing_one_hot = [col for col in cols_for_one_hot if col in df_encoded.columns and col != target_col]

    if existing_one_hot:
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=existing_one_hot,
            drop_first=False,
            dtype=int
        )

    return df_encoded


df_cleaned = clean_and_aggregate_features(df_filtered)

# handling classification of uncommon classes
if hasattr(config_file_tree, 'TASK') and config_file_tree.TASK == 'classification':
    target_col = config_file_tree.TARGET_COL
    if target_col in df_cleaned.columns:
        min_samples = len(df_cleaned) * 0.05  # 5% threshold
        before = df_cleaned.copy()
        df_cleaned = pd.DataFrame(df_cleaned.groupby(target_col).filter(lambda x: len(x) >= min_samples))
        removed_values = set(before[target_col].unique()) - set(df_cleaned[target_col].unique())
        if removed_values:
            print(f"Removed target classes with <5% of samples: {removed_values}")

    MIN_SAMPLES = 20
    counts = df_cleaned['tissue_group'].value_counts()
    valid_tissues = counts[counts >= MIN_SAMPLES].index.tolist()

    df_target = df_cleaned[df_cleaned['tissue_group'].isin(valid_tissues)].copy()

    label_encoder = LabelEncoder()  # to numbers
    df_target[target_col] = label_encoder.fit_transform(df_target['tissue_group'])

    # Mapping (es. {'Adn': 0, 'Asc': 1, 'Mes': 2, 'Ome': 3, 'Oth': 4, 'Ova': 5, 'Per': 6, 'Tub': 7})
    class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Target class mapping:", class_mapping)

    df_processed = encode_categorical_features(df_target)


print("\nFinal shape of processed DataFrame:", df_processed.shape)
print("\nFinal DataFrame columns:")
print(df_processed.columns.tolist())

df_processed.to_csv("preprocessed_clinical.tsv", sep='\t', index=False)




#TODO adattare a nuovi dati clinici:
'''
# removal of patients that had radiation therapy (heavily impacts CNVs)
df_clinical = df_clinical[~df_clinical['radiation_therapy'].fillna('').astype(str).str.contains('YES', case=False)]

features_to_keep = [
    "disease_code",
    "bcr_patient_barcode",
    "vital_status",
    "age_at_initial_pathologic_diagnosis",
    #"race",
    #"ethnicity",
    #"batch_number",
    #"tissue_source_site",
    "clinical_stage",
    "neoplasm_histologic_grade",
    "days_to_last_followup",
    "days_to_death",
    "tumor_residual_disease",
    "anatomic_neoplasm_subdivision ",
    "initial_pathologic_diagnosis_method",
    "primary_therapy_outcome_success",
    "drug_name",
    "days_to_drug_therapy_start",
    "days_to_drug_therapy_end",
    "number_cycles",
    "regimen_indication",
    "therapy_type",
    "therapy_ongoing",
    "postoperative_rx_tx",
    "person_neoplasm_cancer_status",
    "days_to_new_tumor_event_after_initial_treatment",
    "new_neoplasm_event_type",
    "progression_determined_by",
    "lymphatic_invasion",
    "venous_invasion",
]

# filter dataset to available desired features
available_features = [col for col in features_to_keep if col in df_clinical.columns]
df_clinical = df_clinical[available_features]

# exclude features with < 50% valid values
threshold = 0.50 * len(df_clinical)
df_filtered = df_clinical.dropna(thresh=threshold, axis=1)

dropped_cols = set(df_clinical.columns) - set(df_filtered.columns)
if dropped_cols:
    print(f"Dropped due to >50% missing values: {list(dropped_cols)}")


# TODO eventually modify for temporal history (es. values up-to-metastasis), now overall aggregation
def clean_and_aggregate_clinical_data(df):
    """ Clean cols data_HERCULES_DECIDER formatted as lists for every patient. Returns the df with cols replaced with single values or lists for later encoding. """

    df_clean = df.copy()

    if 'vital_status' in df_clean.columns:
        df_clean['vital_status'] = df_clean['vital_status'].apply(clean_vital_status)

    if 'days_to_last_followup' in df_clean.columns:
        df_clean['days_to_last_followup'] = df_clean['days_to_last_followup'].apply(clean_days_to_last_followup)

    if 'days_to_death' in df_clean.columns:
        df_clean['days_to_death'] = df_clean['days_to_death'].apply(clean_days_to_death)

    if 'postoperative_rx_tx' in df_clean.columns:
        df_clean['postoperative_rx_tx'] = df_clean['postoperative_rx_tx'].apply(clean_postoperative_rx_tx)

    if 'primary_therapy_outcome_success' in df_clean.columns:
        df_clean['primary_therapy_outcome_success'] = df_clean['primary_therapy_outcome_success'].apply(clean_primary_therapy_outcome_success)

    if 'number_cycles' in df_clean.columns:
        df_clean['number_cycles'] = df_clean['number_cycles'].apply(clean_number_cycles)

    if all(col in df_clean.columns for col in ['days_to_drug_therapy_start', 'days_to_drug_therapy_end']):
        df_clean['total_drug_therapy_duration_days'] = df_clean.apply(compute_therapy_duration, axis=1)
        df_clean = df_clean.drop(columns=['days_to_drug_therapy_start', 'days_to_drug_therapy_end'])

    if 'person_neoplasm_cancer_status' in df_clean.columns:
        df_clean['person_neoplasm_cancer_status'] = df_clean['person_neoplasm_cancer_status'].apply(clean_person_neoplasm_cancer_status)

    if 'days_to_new_tumor_event_after_initial_treatment' in df_clean.columns:
        df_clean['days_to_new_tumor_event_after_initial_treatment'] = df_clean['days_to_new_tumor_event_after_initial_treatment'].apply(clean_days_to_new_tumor_event_after_initial_treatment)

    if 'regimen_indication' in df_clean.columns:
        df_clean['had_progression_therapy'] = df_clean['regimen_indication'].apply(compute_if_therapy_progression)
        df_clean = df_clean.drop(columns=['regimen_indication'])

    if 'therapy_ongoing' in df_clean.columns:
        df_clean['therapy_ongoing'] = df_clean['therapy_ongoing'].apply(clean_therapy_ongoing)


    """"" features to be split and kept al list/set (for later Multi-Label Binarization) """""

    if 'therapy_type' in df_clean.columns:
        df_clean['therapy_type'] = df_clean['therapy_type'].apply(clean_therapy_type)

    if 'new_neoplasm_event_type' in df_clean.columns:
        df_clean['new_neoplasm_event_type'] = df_clean['new_neoplasm_event_type'].apply(parse_list)

    if 'progression_determined_by' in df_clean.columns:
        df_clean['progression_determined_by'] = df_clean['progression_determined_by'].apply(clean_progression_determined_by)

    if 'drug_name' in df_clean.columns:
        df_clean['drug_name'] = df_clean['drug_name'].apply(clean_drug_name)

    return df_clean


def map_to_num_and_fill_static(df_clean):
    """ Returns features mapped to numerical or static labels/fill_nan for later encoding and nan imputation.
    Applies also static rules to try and infer some features from others.
    Unknowns mapping become nan. """

    df_mapped = df_clean.copy()

    if "disease_code" in df_mapped.columns:  # TODO automatizza
        disease_code_mapping = {
            "LUAD": 0,
            "LUSC": 1,
        }
        df_mapped["disease_code"] = df_mapped["disease_code"].map(disease_code_mapping)


    if "vital_status" in df_mapped.columns:
        vital_status_mapping = {
            "Alive": 0,
            "Dead": 1,
        }
        df_mapped["vital_status"] = df_mapped["vital_status"].map(vital_status_mapping)
        df_mapped = infer_missing_vital_status(df_mapped)
        df_mapped = df_mapped.rename(columns={'vital_status': 'deceased'})

    if "clinical_stage" in df_mapped.columns:
        stage_mapping = {
            "Stage I": 1, "Stage IA": 1, "Stage IB": 2, "Stage IC": 3,
            "Stage II": 4, "Stage IIA": 4, "Stage IIB": 5, "Stage IIC": 6,
            "Stage III": 7, "Stage IIIA": 7, "Stage IIIB": 8, "Stage IIIC": 9,
            "Stage IV": 10,
        }
        df_mapped["clinical_stage"] = df_mapped["clinical_stage"].map(stage_mapping)

    if "neoplasm_histologic_grade" in df_mapped.columns:
        grade_mapping = {
            "G1": 1, "G2": 2,
            "G3": 3, "G4": 4,
        }
        df_mapped["neoplasm_histologic_grade"] = df_mapped["neoplasm_histologic_grade"].map(grade_mapping)

    if "tumor_residual_disease" in df_mapped.columns:
        tumor_residual_disease_mapping = {
            "No Macroscopic disease": 0,
            "1-10 mm": 1,
            "11-20 mm": 2,
            ">20 mm": 3,
        }
        df_mapped["tumor_residual_disease"] = df_mapped["tumor_residual_disease"].map(
            tumor_residual_disease_mapping)

    if "initial_pathologic_diagnosis_method" in df_mapped.columns:
        initial_pathologic_diagnosis_method_mapping = {
            "Tumor resection": "Resection",
            "Cytology (e.g. Peritoneal or pleural fluid)": "Cytology",
        }
        df_mapped["initial_pathologic_diagnosis_method"] = df_mapped["initial_pathologic_diagnosis_method"].map(
            initial_pathologic_diagnosis_method_mapping)
        df_mapped["initial_pathologic_diagnosis_method"] = df_mapped["initial_pathologic_diagnosis_method"].fillna(
            "Biopsy/Other")

    if 'person_neoplasm_cancer_status' in df_mapped.columns:
        df_mapped['person_neoplasm_cancer_status'] = df_mapped['person_neoplasm_cancer_status'].fillna('UNKNOWN')

    if 'primary_therapy_outcome_success' in df_mapped.columns:
        df_mapped['primary_therapy_outcome_success'] = df_mapped.apply(infer_missing_primary_therapy_outcome_success, axis=1)
        df_mapped = df_mapped.rename(columns={'primary_therapy_outcome_success': 'primary_therapy_resistance'})

    if 'had_progression_therapy' in df_mapped.columns:
        df_mapped = infer_had_progression_therapy(df_mapped)


    """"" days_to_death and days_to_last followup unified in --> Overall Survival """""

    if all(col in df_mapped.columns for col in ['deceased', 'days_to_death', 'days_to_last_followup']):
        df_mapped['overall_survival_days'] = np.where(
            df_mapped['deceased'] == 1,
            df_mapped['days_to_death'],
            df_mapped['days_to_last_followup']
        ).astype(float)
        df_mapped = df_mapped.drop(columns=['days_to_death', 'days_to_last_followup'])
        df_mapped = df_mapped.drop(df_mapped[df_mapped['overall_survival_days'] < 0].index)  # to be sure

    return df_mapped


def features_encoding(df_mapped):
    """ Returns categorical features encoded. """
    df_encoded = df_mapped.copy()

    """"" MultiLabelBinarizer for lists of values """""

    cols_for_mlb = ['therapy_type', 'new_neoplasm_event_type', 'progression_determined_by', 'drug_name', ]

    for col in cols_for_mlb:
        if col in df_encoded.columns:

            mlb = MultiLabelBinarizer()
            multi_hot_matrix = mlb.fit_transform(df_encoded[col])

            # new df with cols "col_Value"
            new_df = pd.DataFrame(
                multi_hot_matrix,
                columns=[f"{col}_{cls.replace(' ', '_')}" for cls in mlb.classes_],
                index=df_encoded.index
            )
            df_encoded = (pd.concat([df_encoded, new_df], axis=1))
            df_encoded = df_encoded.drop(columns=col)


    """"" One-Hot Encoding for mutually exclusive features """""

    cols_for_one_hot = ['initial_pathologic_diagnosis_method', 'person_neoplasm_cancer_status',]
    existing_cols = [col for col in cols_for_one_hot if col in df_encoded.columns]

    if existing_cols:
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=existing_cols,
            drop_first=False,
            dtype=int
        )

    return df_encoded


df_cleaned = clean_and_aggregate_clinical_data(df_filtered)
df_remapped = map_to_num_and_fill_static(df_cleaned)
df_encoded = features_encoding(df_remapped)

# convert float cols in integers (for label usage; feats will later be converted to floats and imputed for tensor)
df_processed = df_encoded.fillna(-1)
float_cols = df_processed.select_dtypes(include=['float']).columns
df_processed[float_cols] = df_processed[float_cols].astype(int)


""""" if classification, filter out patients with rare value in target class """""

if config_file_tree.TASK == 'classification':
    target_col = config_file_tree.TARGET_COL
    min_samples = len(df_processed) * 0.10  # threshold with respect to patients total number

    before = df_processed.copy()

    df_processed = pd.DataFrame(df_processed.groupby(target_col).filter(lambda x: len(x) >= min_samples))

    removed_values = set(before[target_col].unique()) - set(df_processed[target_col].unique())

    if removed_values:
        print(f"Removed target classes appearing in <10% of patients: {removed_values}")

print("\nFinal Processed Data Shape:", df_processed.shape)
print("\nColumns in final DataFrame:")
print(df_processed.columns.tolist())

# Save preprocessed clinical data_HERCULES_DECIDER
df_processed.to_csv("preprocessed_clinical_data.tsv", sep='\t', index=False)
'''
