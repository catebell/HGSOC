import re
from os import path
import pandas as pd

import config_file_tree

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
metadata_filepath = path.join(extracted_dir, "samples_metadata.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
clinical_filepath = path.join(output_dir, "preprocessed_clinical_data.tsv")
output_metadata_filepath = path.join(output_dir, "preprocessed_samples_metadata.tsv")


def simplify_tissue(tissue_str):
    """
    Tissues code cleaned of laterality (L/R) and numerical indices.
    E.g.: OvaR1 -> Ova, Ome1 -> Ome, TubR2 -> Tub, Asc1 -> Asc, LN1 -> LN.
    Then maps to 3 anatomical groups:
    1) Solid_Pelvic: Ova, Tub, Adn, Ute, Vag
    2) Solid_Metastasis: Ome, Per, Mes, Bow, LN, Oth
    3) Fluid: Asc, Plf
    """
    if pd.isna(tissue_str):
        return "Unknown", 'Unknown'

    tissue_str = str(tissue_str).strip()

    # based on documentation
    solid_pelvic     = ['Ova', 'Tub', 'Adn', 'Ute', 'Vag']
    fluids = ['Asc', 'Plf']
    solid_metastases = ['Ome', 'Per', 'Mes', 'Bow', 'LN', 'Oth']

    for t in solid_pelvic:
        if tissue_str.lower().startswith(t.lower()):
            return t, 'Solid_Pelvic'

    for t in fluids:
        if tissue_str.lower().startswith(t.lower()):
            return t, 'Fluid'

    for t in solid_metastases:
        if tissue_str.lower().startswith(t.lower()):
            return t, 'Solid_Metastasis'

    if tissue_str.lower().startswith('tum'):  # generic tumor case
        return 'Oth', 'Solid_Metastasis'

    return tissue_str, 'Solid_Metastasis'  # fallback


def preprocess_metadata():
    print(f"Loading metadata from: {metadata_filepath}")
    df_metadata = pd.read_csv(metadata_filepath, sep="\t")
    print(f"Initial metadata shape: {df_metadata.shape}")

    if 'tissue' in df_metadata.columns:
        if 'tissue' in df_metadata.columns:
            tissue_tuple_series = df_metadata['tissue'].apply(simplify_tissue)
            df_metadata[['tissue', 'tissue_type']] = pd.DataFrame(
                tissue_tuple_series.tolist(),
                index=df_metadata.index
            )

    if 'phase_label' in df_metadata.columns:
        df_metadata['phase_label'] = df_metadata['phase_label'].fillna('Unknown')

    if 'sample_class' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['sample_class'] == 'BulkTissue']
        print(f"Discarded {n_before - len(df_metadata)} samples with sample_class != 'BulkTissue' (CellLines, OrganoidLines, ...).")

    if 'treatment' in df_metadata.columns:
        n_before = len(df_metadata)
        df_metadata = df_metadata[df_metadata['treatment'] == 'Untreated']
        print(f"Discarded {n_before - len(df_metadata)} samples with treatment != 'Untreated'.")

    if 'extraction' in df_metadata.columns:
        n_before = len(df_metadata)
        forbidden_pattern = r'set|dup'
        has_suffix = df_metadata['extraction'].astype(str).str.contains(forbidden_pattern, case=False, na=False)
        df_metadata = df_metadata[~has_suffix]
        print(f"Discarded {n_before - len(df_metadata)} samples with suffixes _set or _dup.")

        # keep only updated versioning: _Data(number)
        n_before = len(df_metadata)
        has_data = df_metadata['extraction'].astype(str).str.contains(r'Data\d+', case=False, na=False)
        data_sample_ids = df_metadata[has_data]['sample_id']
        # sample_id without 'Data(number)' (es. 'H001_pOvaR1_RNA1_Data2' -> 'H001_pOvaR1_RNA1')
        base_sample_ids_to_drop = set(
            data_sample_ids.apply(lambda s: re.sub(r'(_)?Data\d+', '', str(s), flags=re.IGNORECASE))
        )
        if base_sample_ids_to_drop:
            df_metadata = df_metadata[~df_metadata['sample_id'].isin(base_sample_ids_to_drop)]
        print(
            f"Discarded {n_before - len(df_metadata)} samples by keeping updated sequencing (_Data2).")


    features_to_keep = [
        "sample_id",
        "sample",
        "patient_id",
        "cohort",
        "phase",
        "phase_label",
        "tissue",
        "tissue_type",
        "sample_class",
        "extraction",
        "treatment"
    ]

    available_features = [col for col in features_to_keep if col in df_metadata.columns]
    df_clean = df_metadata[available_features].copy()

    fluid_prefixes = ['Asc', 'Plf']  # based on PCA, it could be useful to signal the GNN that some samples are fluid
    df_clean['is_fluid'] = df_clean['tissue'].astype(str).apply(
        lambda t: 'yes' if any(t.startswith(prefix) for prefix in fluid_prefixes) else 'no'
    )

    print(f"\nLoading patient clinical data from: {clinical_filepath}")
    df_clinical = pd.read_csv(clinical_filepath, sep='\t')

    # filter out samples of patients with no clinical record
    clinical_patients = df_clinical['Patient_cohort_code_Patient_Card']
    n_patients = len(df_clean['patient_id'].unique())
    df_clean = df_clean[df_clean['patient_id'].str.upper().isin(clinical_patients.str.upper())]
    print(f"Discarded {n_patients - len(df_clean['patient_id'].unique())} patients with no clinical record.")

    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    print(f"\nPreprocessed metadata saved to: '{output_metadata_filepath}' (Shape: {df_clean.shape})")


if __name__ == "__main__":
    preprocess_metadata()
    print("\nDONE!")
