import re
from os import path
import pandas as pd
from tqdm import tqdm

import config_file_tree

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
metadata_filepath = path.join(extracted_dir, "samples_metadata.tsv")
deconv_w_filepath = config_file_tree.DECONVOLUTION_W_FILE

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
clinical_filepath = path.join(output_dir, "preprocessed_clinical_data.tsv")
output_metadata_filepath = path.join(output_dir, "preprocessed_samples_metadata.tsv")

EOC_THRESHOLD = 0.30  # minimum tumor purity
COHORT_THRESHOLD = 1  # % of minimum samples per cohort wrt the total num of samples


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

    # samples deduplication (es. re-extractions)
    if 'extraction' in df_metadata.columns:
        n_samples = len(df_metadata)
        forbidden_pattern = r'set|dup'
        has_suffix = df_metadata['extraction'].astype(str).str.contains(forbidden_pattern, case=False, na=False)
        df_metadata = df_metadata[~has_suffix]
        print(f"Discarded {n_samples - len(df_metadata)} samples with suffixes _set or _dup.")

        # keep only updated versioning: _Data(number)
        n_samples = len(df_metadata)
        has_data = df_metadata['extraction'].astype(str).str.contains(r'Data\d+', case=False, na=False)
        data_sample_ids = df_metadata[has_data]['sample_id']

        # sample_id without 'Data(number)' (es. 'H001_pOvaR1_RNA1_Data2' -> 'H001_pOvaR1_RNA1')
        base_sample_ids_to_drop = set(
            data_sample_ids.apply(lambda s: re.sub(r'(_)?Data\d+', '', str(s), flags=re.IGNORECASE))
        )
        if base_sample_ids_to_drop:
            df_metadata = df_metadata[~df_metadata['sample_id'].isin(base_sample_ids_to_drop)]
        print(
            f"Discarded {n_samples - len(df_metadata)} samples by keeping updated sequencing (_Data2).")

    if 'phase_label' in df_metadata.columns:
        df_metadata['phase_label'] = df_metadata['phase_label'].fillna('Unknown')

    if 'tissue' in df_metadata.columns:
        tissue_tuple_series = df_metadata['tissue'].apply(simplify_tissue)
        df_metadata[['tissue', 'tissue_type']] = pd.DataFrame(
            tissue_tuple_series.tolist(),
            index=df_metadata.index
        )

        # add feature
        fluid_prefixes = ['Asc', 'Plf']
        df_metadata['is_fluid'] = df_metadata['tissue'].astype(str).apply(
            lambda t: 'yes' if any(t.startswith(prefix) for prefix in fluid_prefixes) else 'no'
        )


    features_to_keep = [
        "sample_id",
        "sample",
        "patient_id",
        "cohort",
        "phase",
        "phase_label",
        "tissue",
        "tissue_type",
        "is_fluid",
        "sample_class",
        "extraction",
        "treatment"
    ]

    available_features = [col for col in features_to_keep if col in df_metadata.columns]
    df_clean = df_metadata[available_features].copy()

    print(f"\nLoading patient clinical data from: {clinical_filepath}")
    df_clinical = pd.read_csv(clinical_filepath, sep='\t')

    # filter out samples of patients with no clinical record
    clinical_patients = df_clinical['patient_id']
    n_samples = len(df_clean)
    n_patients = len(df_clean['patient_id'].unique())
    df_clean = df_clean[df_clean['patient_id'].str.upper().isin(clinical_patients.str.upper())]
    print(f"Discarded {n_samples - len(df_clean)} samples from {n_patients - len(df_clean['patient_id'].unique())} patients with no clinical record.")


    """"" filter out samples with low tumor purity (EOC) """""

    print(f"\n[FILTERING PURITY] Applying Tumor Purity filter (EOC > {EOC_THRESHOLD * 100:.0f}%)...")
    df_w = pd.read_csv(deconv_w_filepath, sep='\t', usecols=[0, 1], dtype={'EOC': 'float32'})
    sample_col = df_w.columns[0]
    eoc_valid_samples = set(df_w.loc[df_w['EOC'] > EOC_THRESHOLD, sample_col])
    n_samples = len(df_clean)
    df_clean = df_clean[df_clean['sample_id'].isin(eoc_valid_samples)].copy()
    print(f"Discarded {n_samples - len(df_clean)} samples with EOC <= {EOC_THRESHOLD}.")


    """"" filter out cohorts with less than 10% samples of the dataset """

    total_samples = len(df_clean)
    print("Cohorts stats:")
    cohort_stats = []

    for cohort_name, group in tqdm(df_metadata.groupby('cohort')):
        n_samples = len(group)
        percentage = (n_samples / total_samples) * 100
        n_patients = group['patient_id'].nunique() if 'patient_id' in group.columns else 0
        n_tissues = group['tissue'].nunique() if 'tissue' in group.columns else 0

        cohort_stats.append({
            'cohort': cohort_name,
            'n_samples': n_samples,
            'percentage': percentage,
            'n_patients': n_patients,
            'n_tissues': n_tissues
        })

        print(f"Cohort '{cohort_name}':")
        print(f"  - Samples:  {n_samples} ({percentage:.2f}%)")
        print(f"  - Patients: {n_patients}")
        print(f"  - Tissues:  {n_tissues}\n")

    cohorts_to_keep = [s['cohort'] for s in cohort_stats if s['percentage'] >= COHORT_THRESHOLD]
    cohorts_to_remove = [s['cohort'] for s in cohort_stats if s['percentage'] < COHORT_THRESHOLD]

    print(f"[FILTERING COHORT] Kept Cohorts (>= {COHORT_THRESHOLD}%): {cohorts_to_keep}\n")
    if cohorts_to_remove:
        print(f"[FILTERING COHORT] Removed Cohorts (< {COHORT_THRESHOLD}%): {cohorts_to_remove}\n")

    # filtering metadata
    df_clean = df_clean[df_clean['cohort'].isin(cohorts_to_keep)].copy()


    """"" deduplication of multiple RNA extraction """""

    dup_mask = df_clean.duplicated('sample', keep=False)
    num_duplicates = dup_mask.sum()
    print(f"[DEDUPLICATING SAMPLES] Found {num_duplicates} duplicated samples cols (multiple extractions).")

    # re-ordering by 'extraction' (es. 'RNA1' before 'RNA2', 'RNA3', ...) and keeping first
    df_clean = (
        df_clean
        .sort_values(by='extraction')
        .drop_duplicates(subset=['sample'], keep='first')
        .sort_index()
    )


    df_clean.to_csv(output_metadata_filepath, sep='\t', index=False)
    print(f"\nPreprocessed metadata saved to: '{output_metadata_filepath}' (Shape: {df_clean.shape})")


if __name__ == "__main__":
    preprocess_metadata()
    print("\nDONE!")
