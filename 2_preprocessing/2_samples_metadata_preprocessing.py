from os import path, makedirs
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config_file_tree

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
metadata_filepath = path.join(extracted_dir, "raw_samples_metadata.tsv")
clinical_filepath = path.join(preprocessed_dir, "preprocessed_clinical_data.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
makedirs(output_dir, exist_ok=True)

output_metadata_filepath = path.join(output_dir, "preprocessed_samples_metadata.tsv")


def simplify_tissue(tissue_str):
    """
    Tissues code cleaned of laterality (L/R) and numerical indices.
    E.g.: OvaR1 -> Ova, Ome1 -> Ome, TubR2 -> Tub, Asc1 -> Asc, LN1 -> LN.
    Then maps to 3 anatomical groups:
    1) Solid_Pelvic: Ova, Tub, Adn, Ute
    2) Solid_Metastasis: Ome, Per, Mes, Bow, LN, Oth
    3) Fluid: Asc, Plf
    """
    if pd.isna(tissue_str):
        return 'Unknown'

    tissue_str = str(tissue_str).strip()

    cleaned = re.sub(r'[LR]?\d+$', '', tissue_str, flags=re.IGNORECASE)
    cleaned = re.sub(r'[LR]$', '', cleaned, flags=re.IGNORECASE)

    # based on documentation
    solid_pelvic     = ['Ova', 'Tub', 'Adn', 'Ute']
    fluids = ['Asc', 'Plf']
    solid_metastases = ['Ome', 'Per', 'Mes', 'Bow', 'LN', 'Oth']

    for t in solid_pelvic:
        if cleaned.lower() == t.lower() or tissue_str.lower().startswith(t.lower()):
            return 'Solid_Pelvic'

    for t in fluids:
        if cleaned.lower() == t.lower() or tissue_str.lower().startswith(t.lower()):
            return 'Fluid'

    for t in solid_metastases:
        if cleaned.lower() == t.lower() or tissue_str.lower().startswith(t.lower()):
            return 'Solid_Metastasis'

    return 'Solid_Metastasis'  # fallback


def preprocess_metadata():
    print(f"Loading metadata from: {metadata_filepath}")
    df_metadata = pd.read_csv(metadata_filepath, sep="\t")
    print(f"Initial metadata shape: {df_metadata.shape}")

    if 'tissue' in df_metadata.columns:
        df_metadata['tissue'] = df_metadata['tissue'].apply(simplify_tissue)

    if 'phase_label' in df_metadata.columns:
        df_metadata['phase_label'] = df_metadata['phase_label'].fillna('Unknown')

    features_to_keep = [
        "sample_id",
        "patient_id",
        "cohort",
        "phase",
        "phase_label",
        "tissue",
        "extraction"
    ]

    available_features = [col for col in features_to_keep if col in df_metadata.columns]
    df_clean = df_metadata[available_features].copy()

    fluid_prefixes = ['Asc', 'Plf']  # based on PCA, it could be useful to signal the GNN that some samples are fluid
    df_clean['is_fluid'] = df_clean['tissue'].astype(str).apply(
        lambda t: 'yes' if any(t.startswith(prefix) for prefix in fluid_prefixes) else 'no'
    )

    # TODO temporaneo solo per avere i campioni pre-trattametno e non recidive
    if 'phase_label' in df_clean.columns:
        df_clean = df_clean[df_clean['phase_label'] == 'Primary'].copy()
        print(f"Filtered for 'Primary' phase: {len(df_clean)} samples kept.")

    print(f"\nLoading patient clinical data from: {clinical_filepath}")
    df_clinical = pd.read_csv(clinical_filepath, sep='\t')

    # merging with patient clincal data (maybe da non fare se poi gli embeddings verranno creati a due livelli)
    if 'Patient_cohort_code_Patient_Card' in df_clinical.columns:
        df_clinical = df_clinical.rename(columns={'Patient_cohort_code_Patient_Card': 'patient_id'})
    elif df_clinical.index.name == 'Patient_cohort_code_Patient_Card':
        df_clinical = df_clinical.reset_index().rename(columns={'Patient_cohort_code_Patient_Card': 'patient_id'})

    df_clean['patient_id'] = df_clean['patient_id'].astype(str).str.lower().str.strip()
    df_clinical['patient_id'] = df_clinical['patient_id'].astype(str).str.lower().str.strip()

    # keep only samples having associated clinical data
    df_clean = pd.merge(df_clean, df_clinical, on='patient_id', how='inner')
    print(f"Merged samples with clinical data. Shape after merge: {df_clean.shape}")



    """"" filter for eventual rare targets """""

    target_col = config_file_tree.TARGET_COL
    if target_col not in df_clean.columns:
        raise KeyError(f"Chosen target column '{target_col}' does not exists in samples metadata!"
                       f"Available columns: {list(df_clean.columns)}")

    print("Excluding samples where target = nan...")
    df_clean = df_clean.dropna(subset=[target_col]).copy()
    if config_file_tree.TASK == 'survival':
        time_col = config_file_tree.TIME_COL
        df_clean = df_clean[df_clean[time_col] > 0]  # remove non-valid times
        df_clean = df_clean.dropna(subset=[time_col])

    if config_file_tree.TASK == 'classification':
        # drop patients with less than min_patients_per_class occurrences (not stratifiable)
        min_patients_per_class = 3  # 3 = minimum for train/val/test split (split based on patient_id)
        df_patients = df_clean[['patient_id', target_col]].drop_duplicates(subset=['patient_id'], keep='first')
        patient_counts = df_patients.groupby(target_col)['patient_id'].nunique()
        rare_classes = patient_counts[patient_counts < min_patients_per_class].index

        if len(rare_classes) > 0:
            print(f"\nRemoved target classes with < {min_patients_per_class} patients:")
            for cls in rare_classes:
                print(f"  - Class '{cls}': {patient_counts[cls]} patients")
            df_clean = df_clean[~df_clean[target_col].isin(rare_classes)].copy()  # filtered dataset


    '''
    """"" encoding for target column """""

    if config_file_tree.TASK == 'classification':
        label_encoder = LabelEncoder()
        df_clean[target_col] = label_encoder.fit_transform(df_clean[target_col].astype(str))

        class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        print(f"\n" + "=" * 50)
        print(f"TARGET ENCODING MAPPING ('{target_col}'):")
        print("=" * 50)
        for class_name, code in class_mapping.items():
            print(f"  {class_name:<15} -> {code}")
        print("=" * 50 + "\n")

    print("STANDARDIZED TARGET DISTRIBUTION:")
    print(df_clean[target_col].value_counts())
    '''
    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    print(f"\nPreprocessed metadata saved to: '{output_metadata_filepath}' (Shape: {df_clean.shape})")


if __name__ == "__main__":
    preprocess_metadata()
    print("\nDONE!")
