import logging
import re
from os import path, makedirs
from pandas import read_csv, DataFrame
from tqdm import tqdm

import config_file_tree


""" [!] All specific info gathered from project documentation """


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


""""" paths and parameters """""

dataset_dir = config_file_tree.DATASET_DIR
RNA_bulk_raw = config_file_tree.RNA_BULK_RAW_FILE

makedirs(dataset_dir, exist_ok=True)

out_metadata_filepath = path.join(dataset_dir, "samples_metadata.tsv")


def parse_sample_id(sample_id):
    """
    Decode naming convention (es. H001_pOvaR1_FS1_RNA1 -> Patient, Phase, Tissue, Extraction)
    """

    tokens = sample_id.split('_')

    info = {
        'sample_id': sample_id,
        'cohort': 'Unknown',
        'patient_id': tokens[0],
        'phase': None,
        'phase_label': 'Unknown',
        'tissue': None,
        'sample_class': 'BulkTissue',
        'extraction': None
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


    #  remaining tokens: Clinical phase + Tissue (es. pOvaR, rLN1, iAsc1, p0me1) and/or Subsample + Extraction

    extraction_tokens = []

    for token in tokens[1:]:
        # extraction data (es. RNA1, DNA2, Data1)
        if re.match(r"^(RNA|DNA|Data)\d*$", token, re.IGNORECASE):
            extraction_tokens.append(token)
            continue

        # subsample type / cellular line (es. CLadh, CLsph, CL62, FS1, PE, DE, DT)
        m_sub = re.match(r"^(?P<subsample>CL\w*|FS\d*|PE\w*|DE|DT|SCB|SC|HE|PAX)$", token, re.IGNORECASE)
        if m_sub:
            #info['subsample'] = token
            sub_upper = token.upper()
            if sub_upper.startswith('CL'):
                info['sample_class'] = 'CellLine'
            elif sub_upper.startswith('FS'):
                info['sample_class'] = 'FreshFrozen'
            elif any(sub_upper.startswith(x) for x in ['PE', 'DE', 'HE', 'PAX']):
                info['sample_class'] = 'FFPE'
            elif sub_upper == 'DT':
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

    if extraction_tokens:
        info['extraction'] = "_".join(extraction_tokens)

    return info


def get_hercules_metadata():
    logging.info(f"Loading matrix (genes x samples) from {RNA_bulk_raw.split(path.sep)[-1]}...")
    df_bulk = read_csv(RNA_bulk_raw, sep="\t")

    sample_cols = list(df_bulk.columns[1:])

    logging.info("Metadata extraction from samples IDs...")
    metadata_list = []
    for sample_id in tqdm(sample_cols):
        parsed_info = parse_sample_id(sample_id)
        parsed_info['sample_id'] = sample_id
        metadata_list.append(parsed_info)

    df_metadata = DataFrame(metadata_list)

    df_metadata.to_csv(out_metadata_filepath, index=False, sep='\t')
    logging.info(f"Metadata saved in: {out_metadata_filepath}\n")


if __name__ == "__main__":
    get_hercules_metadata()
    logging.info("DONE!\n\n")