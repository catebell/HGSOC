from os import path, makedirs
import re
import pandas as pd
import config_file_tree

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
metadata_filepath = path.join(extracted_dir, "extracted_samples_metadata.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
makedirs(output_dir, exist_ok=True)

output_metadata_filepath = path.join(output_dir, "preprocessed_samples_metadata.tsv")


def simplify_tissue(tissue_str):
    """
    Tissues code cleaned of laterality (L/R) and numerical indices.
    E.g.: OvaR1 -> Ova, Ome1 -> Ome, TubR2 -> Tub, Asc1 -> Asc, LN1 -> LN.
    """
    if pd.isna(tissue_str):
        return 'Unknown'

    tissue_str = str(tissue_str).strip()

    cleaned = re.sub(r'[LR]?\d+$', '', tissue_str, flags=re.IGNORECASE)
    cleaned = re.sub(r'[LR]$', '', cleaned, flags=re.IGNORECASE)

    # based on documentation
    known_tissues = ['Ome', 'Ova', 'Per', 'Asc', 'Tub', 'Ute', 'LN', 'Bow', 'Plf', 'Mes', 'Adn', 'Oth']

    for t in known_tissues:
        if cleaned.lower() == t.lower() or tissue_str.lower().startswith(t.lower()):
            return t

    return 'Other'


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
        lambda t: 1 if any(t.startswith(prefix) for prefix in fluid_prefixes) else 0
    )


    """"" filter for eventual rare targets """""

    if config_file_tree.TASK == 'classification':
        target_col = config_file_tree.TARGET_COL
        cols_to_fetch = [target_col]
    else:
        target_col = config_file_tree.SURVIVAL_EVENT_COL
        time_col = config_file_tree.SURVIVAL_TIME_COL
        df_clean = df_clean[df_clean[time_col] > 0]  # remove non-valid times
        df_clean[target_col] = df_clean[target_col].astype(int)
        cols_to_fetch = [target_col, time_col]

    # drop nan in target col (not useful)
    df_clean = df_clean.dropna(subset=cols_to_fetch).copy()

    min_patients_per_class = 3  # 3 = minimum for train/val/test split (split based on patient_id)

    # drop patients with less than min_patients_per_class occurrences (not stratifiable)
    df_patients = df_clean[['patient_id', target_col]].drop_duplicates(subset=['patient_id'], keep='first')
    patient_counts = df_patients.groupby(target_col)['patient_id'].nunique()
    rare_classes = patient_counts[patient_counts < min_patients_per_class].index

    if len(rare_classes) > 0:
        print(f"\nRemoved target classes with < {min_patients_per_class} patients:")
        for cls in rare_classes:
            print(f"  - Class '{cls}': {patient_counts[cls]} patients")
        df_clean = df_clean[~df_clean[target_col].isin(rare_classes)].copy()  # filtered dataset


    print("STANDARDIZED TISSUE DISTRIBUTION:")
    print(df_clean['tissue'].value_counts())

    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    print(f"\nPreprocessed metadata saved to: '{output_metadata_filepath}' (Shape: {df_clean.shape})")


if __name__ == "__main__":
    preprocess_metadata()
    print("\nDONE!")
