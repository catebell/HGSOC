import logging
import re
from os import path, makedirs
from pathlib import Path

from pandas import read_csv, DataFrame, concat
from tqdm import tqdm

import PCA
import config_file_tree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

""""" paths and parameters """""

dataset_bulk_expr_dir = config_file_tree.ORIGINAL_RNA_DIR

metadata_filepath = path.join(config_file_tree.DATASET_DIR, "samples_metadata.tsv")

output_dir = config_file_tree.EXTRACTED_DATA_DIR
output_metadata_filepath = path.join(output_dir, "samples_metadata.tsv")
output_bulk_filepath = path.join(output_dir, "raw_bulk_expr.tsv")


def load_and_merge_tpm_tables(
    base_dir: str = dataset_bulk_expr_dir, target_filename: str = "TPMlog2+1_gene_table.tsv") -> DataFrame:

    """ Searches recursively TPMlog2+1_gene_table.tsv files in the dir provided, cleaning Ensembl IDs.
    Returns a single unified matrix (Genes x Samples). """

    base_path = Path(base_dir)
    logging.info(
        f"Retrieving files '{target_filename}' in: {base_path.resolve()}..."
    )

    matching_files = sorted(list(base_path.glob(f"**/{target_filename}")))

    if not matching_files:
        raise FileNotFoundError(
            f"No file '{target_filename}' found in {base_dir}."
        )

    logging.info(f"Found {len(matching_files)} files.")

    dfs = []
    for fpath in tqdm(matching_files, desc="Loading TPM tables"):
        df = read_csv(fpath, sep="\t", index_col=0)

        # Ensembl ID cleaned (es. ENSG00000000003.16 -> ENSG00000000003)
        df.index = df.index.astype(str).str.split(".").str[0]

        if df.index.has_duplicates:
            df = df.groupby(df.index).max()  # max expression in case of duplicates

        dfs.append(df)

    logging.info("Merging tables by columns...")
    merged_bulk = DataFrame(concat(dfs, axis=1)).copy()

    if merged_bulk.columns.duplicated().any():
        n_dup = merged_bulk.columns.duplicated().sum()
        logging.warning(
            f"Found {n_dup} duplicated samples, keeping first occurrence."
        )
        merged_bulk = merged_bulk.loc[:, ~merged_bulk.columns.duplicated()]

    merged_bulk = merged_bulk.fillna(0.0)

    merged_bulk = merged_bulk.reset_index()
    merged_bulk.rename(columns={merged_bulk.columns[0]: "gene_id"}, inplace=True)

    logging.info(
        f"Raw matrix created: {merged_bulk.shape[0]} genes x {merged_bulk.shape[1]-1} samples. Saving to {output_bulk_filepath}..."
    )

    merged_bulk.to_csv(output_bulk_filepath, sep="\t", index=False)
    return merged_bulk


def parse_sample_id(sample_id):
    """
    Decode naming convention (es. H001_pOvaR1_FS1_RNA1 -> Patient, Phase, Tissue, Extraction)
    """

    tokens = sample_id.split('_')

    info = {
        'sample_id': sample_id,
        'sample': '_'.join(tokens[:2]),
        'cohort': 'Unknown',
        'patient_id': tokens[0],
        'phase': None,
        'phase_label': 'Unknown',
        'tissue': None,
        'sample_class': 'BulkTissue',
        'extraction': None,
        'treatment': 'Untreated',
    }

    if not tokens:
        return info


    # first token: Cohort + Patient Number (es. OC038, H002, D365)

    patient_token = tokens[0]
    match_patient = re.match(r"^(?P<cohort>H|D|M|OC|SYKO)(?P<num>\d+)$", patient_token, re.IGNORECASE)
    if match_patient:
        info['cohort'] = match_patient.group('cohort').upper()
        info['patient_id'] = patient_token
    elif patient_token.lower().startswith('ovcar') or 'cl' in patient_token.lower():
        info['cohort'] = 'CellLine_Control'
        info['patient_id'] = patient_token
        info['sample_class'] = 'CellLine'
    else:
        match_generic = re.match(r"^(?P<cohort>[A-Za-z]+)(?P<num>\d*)$", patient_token)
        if match_generic:
            info['cohort'] = match_generic.group('cohort').upper()

    if len(tokens) == 1:
        return info


    #  remaining tokens: Clinical phase + Tissue (es. pOvaR, rLN1, iAsc1, p0me1) and/or Subsample + Extraction + Treatment

    extraction_tokens = []

    for token in tokens[1:]:
        # extraction data (es. RNA1, DNA2, Data1)
        if re.match(r"^(RNA|DNA|Data|dup|set)\d*$", token, re.IGNORECASE):
            extraction_tokens.append(token)
            continue

        # subsample type / cellular line (es. CLadh, CLsph, CL62, FS1, PE, DE, DT)
        m_sub = re.match(r"^(?P<subsample>CL\w*|OL\d*|FS\d*|PE\w*|DE|DT|SCB|SC|HE|PAX)$", token, re.IGNORECASE)
        if m_sub:
            #info['subsample'] = token
            if token.upper().startswith('CL'):
                info['sample_class'] = 'CellLine'
            if token.upper().startswith('OL'):
                info['sample_class'] = 'OrganoidLine'  # Organoid line indicator (es. OL2184, OL2171)
            elif token.upper().startswith('FS'):
                info['sample_class'] = 'FreshFrozen'
            elif any(token.upper().startswith(x) for x in ['PE', 'DE', 'HE', 'PAX']):
                info['sample_class'] = 'FFPE'
            elif token.upper() == 'DT':
                info['sample_class'] = 'DissociatedTissue'
            continue

        # clinical phase + tissue (es. pOvaR, pPer, rLN1, iOme1)
        match_phase_tissue = re.match(r"^(?P<phase>[piro]\d*)(?P<tissue>[A-Z][A-Za-z0-9]*)$", token)
        if match_phase_tissue:
            info['phase'] = match_phase_tissue.group('phase')
            info['tissue'] = match_phase_tissue.group('tissue')

            phase_code = info['phase'][0].lower() if info['phase'] else ''
            phase_map = {'p': 'Primary', 'i': 'Interval', 'r': 'Relapse', 'o': 'Other'}
            info['phase_label'] = phase_map.get(phase_code, 'Unknown')
            continue

        # tissue with no phase (es. TubR2, TubL, OvaR)
        match_tissue_only = re.match(r"^(?P<tissue>[A-Z][A-Za-z0-9]*)$", token)
        if match_tissue_only and info['tissue'] is None:
            info['tissue'] = token
            continue

        known_treatments = ['Alpelisib', 'DMSO']
        if token in known_treatments:
            info['treatment'] = token
        elif token == 'TR':
            info['treatment'] = 'Treated'


    if extraction_tokens:
        info['extraction'] = "_".join(extraction_tokens)

    return info


def extract_hercules_metadata(df_bulk):
    """ [!] All specific info gathered from project documentation """

    logging.info("Metadata extraction from samples IDs...")
    sample_cols = list(df_bulk.columns[1:])

    metadata_list = []
    for sample_id in tqdm(sample_cols):
        parsed_info = parse_sample_id(sample_id)
        parsed_info['sample_id'] = sample_id
        metadata_list.append(parsed_info)

    df_metadata = DataFrame(metadata_list)

    df_metadata.to_csv(output_metadata_filepath, index=False, sep='\t')
    logging.info(f"Metadata saved in: {output_metadata_filepath}\n")


if __name__ == "__main__":

    expr_df = load_and_merge_tpm_tables()

    extract_hercules_metadata(expr_df)

    if config_file_tree.RUN_PCA:
        out_dir_pca = path.join(output_dir, "out_analysis")
        makedirs(out_dir_pca, exist_ok=True)

        PCA.run_pca_analysis(output_bulk_filepath, output_metadata_filepath, out_dir_pca)
    print("\nDONE!")