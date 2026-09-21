import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def correct_histology(row):
    re_eval = row['Histology_re_evaluated_notes']  # if there's one, replace Histology value

    if pd.notna(re_eval) and str(re_eval).strip() != '':
        val = str(re_eval).lower().strip()

        if 'hgsc' in val or 'high grade' in val:
            return 'high grade serous'
        elif 'lgsc' in val or 'low grade' in val:
            return 'low grade serous'
        elif 'endometr' in val:
            return 'endometrioid'
        elif 'carcinosarcoma' in val:
            return 'carcinosarcoma'

        return val

    primary = row['Histology']
    if pd.notna(primary):
        return str(primary).lower().strip()

    return 'unknown'


def combine_oper2_cols(row):
    """
    - 'not_applicable_pds': interval surgery not planned
    - 'ids_canceled': interval surgery canceled
    - 'ids_debulked': debulking surgery successfully done
    - 'ids_exploratory_only': no debulking
    - 'ids_not_debulked': surgery failed or not done
    """
    strat = str(row.get('Treatment_strategy', '')).strip()
    cancelled = str(row.get('Oper2_IDS_cancelled', '')).strip()
    lap_eval = str(row.get('Oper2_laparoscopic_evaluation', '')).strip()
    lap_expl = str(row.get('Oper2_explorative_laparotomy', '')).strip()
    debulk = str(row.get('Oper2_debulking_surgery', '')).strip()

    if strat == 'pds' or (pd.isna(row.get('Oper2_debulking_surgery')) and pd.isna(row.get('Oper2_IDS_cancelled'))):
        return 'not_applicable_pds'

    if cancelled == 'yes':
        return 'ids_cancelled'

    if debulk == 'yes':
        return 'ids_debulked'

    if debulk == 'no':
        if lap_eval == 'yes' or lap_expl == 'yes':
            return 'ids_exploratory_only'
        else:
            return 'ids_not_debulked'

    return 'unknown'


def clean_and_infer_chemotherapy_features(df_in):
    """ Clean, infer missing values and creates new features:

    1) Converts columns in numerical values so invalid data = nan
    2) Inference: if Treatment_strategy = PDS, NACT_cycles = 0 and Post_IDS_cycles = 0
    """

    col_primary = "Primary_chemotherapy_cycles"
    col_nact = "NACT_cycles"
    col_post_ids = "Post_IDS_chemotherapy_cycles"
    col_strategy = "Treatment_strategy"

    df_out = df_in.copy()

    chemo_cols = [col_primary, col_nact, col_post_ids]
    for col in chemo_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    if col_strategy in df_out.columns:
        pds_mask = (df_out[col_strategy].astype(str).str.strip().str.upper() == "PDS")

        if col_nact in df_out.columns:
            df_out.loc[pds_mask & df_out[col_nact].isna(), col_nact] = 0.0

        if col_post_ids in df_out.columns:
            df_out.loc[pds_mask & df_out[col_post_ids].isna(), col_post_ids] = 0.0

    return df_out


def clean_residual_tumor(val):
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    if 'more than 10' in val_str or '>10' in val_str:
        return '>10mm'
    elif '1-10' in val_str:
        return '1-10mm'
    elif '0' in val_str:
        return '0'
    else:
        return np.nan


def clean_maintenance_therapy(val):
    """ Discards wrong text, maps synonyms (es. 'beva' -> 'bevacizumab') and returns a list. """
    if not isinstance(val, str) or not val.strip():
        return []

    val = val.strip()

    if val in ["no", "nd", "none", "nan"] or re.match(
        r"^\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}$", val
    ):
        return []

    drug_patterns = [
        (r"bevacizumab|beva", "anti_vegf"),
        (r"niraparib|olaparib", "parp_inhibitor"),
        (r"durvalumab|atezolizumab|dostarlimab|tsr-?042|immunologic", "immunotherapy"),
    ]

    found_drugs = set()
    for pattern, std_name in drug_patterns:
        if re.search(pattern, val):
            found_drugs.add(std_name)

    return sorted(list(found_drugs))


def map_primary_outcome_binary(val):
    """
    For later binarization
    - 0 (Non-Responder): Chemio-resistant or stable disease/progressive disease
    - 1 (Responder) (complete response / partial response)
    - NaN: not defined
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    if 'response' in val_str:
        return 'responsive'
    elif 'disease' in val_str:
        return 'unresponsive'
    else:
        return np.nan


def clean_survival(val):
    """
    - event_os: target event for later Overall Survival (1 = deceduto per qualsiasi causa, 0 = alive/censored)
    - Survival --> Disease-Specific Survival event (1 = deceased by cancer, 0 = alive or different death cause/censored)
    """

    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()

    # if val_str == 'alive':
    #    return 'alive'
    # elif 'cancer' in val_str:
    #    return 'dead_due_to_cancer'
    # elif 'other' in val_str:
    #    return 'dead_due_to_other_reason'
    # elif 'unknown' in val_str:
    #    return 'dead_reason_unknown'
    # return 'unknown'

    if 'alive' in val_str:
        return 'alive'
    elif 'cancer' in val_str:
        return 'dead'  # deceased because of cancer
    return np.nan


def process_survival_time_features(df):
    """
    - Unifies temporal cols in intervals (OS, PFS e PFI)
    - Computes PFI classification target: 1 if Platinum resistant (PFI < 180 days), 0 else (Platinum sensible)
    """
    df = df.copy()

    # 1. Overall Survival Days (OS)
    df['OS_days'] = df['OS_overall_survival_Days_from_Dg_to_death'].fillna(
        df['OS_current_Time_from_dg_alive__Days']
    )

    # 2. Progression-Free Survival Days (PFS)
    df['PFS_days'] = df['PFS_Days_from_Dg_to_1st_prog'].fillna(
        df['PFS_Current_no_prog_alive_days']
    )

    # 3. Platinum-Free Interval Days (PFI)
    df['PFI_days'] = df['PFI_after_Primary_therapy_if_Prog_Days'].fillna(
        df['PFI_current_no_prog_alive_days']
    )

    df['PFI_days'] = df['PFI_days'].apply(lambda x: np.nan if x < 0 else x)  # remove eventual wrong values

    # infer Platinum Chemo-Sensibility
    df['is_platinum_resistant'] = np.where(df['PFI_days'] < 180, 'yes', 'no')
    df.loc[df['PFI_days'].isna(), 'is_platinum_resistant'] = np.nan

    return df


def clean_brca_mutation(row):
    """
    Unifies in single binary col for later binarization:
    - 1 (mutated): BRCA1 or BRCA2 mutation present
    - 0 (wild_type): No mutation registered
    - NaN / 'unknown': not tested or missing data
    """
    stat = str(row.get('BRCA_mutation_status', ''))
    decider = str(row.get('BRCA_any_in_Decider', ''))
    any_flag = str(row.get('BRCA_mutation_any', ''))

    if any_flag == 'yes' or decider == 'yes' or 'brca' in stat and 'no brca' not in stat:
        return 'yes' # ('mutated')

    if decider == 'no' or 'no brca' in stat:
        return 'no' # ('wild_type')

    return np.nan


def clean_hrd_status(df):
    """
    - Integrates HRD/HRP tumor status using 'pretreatment_WGS'
    - Handles missing data as a different category.
    """
    df = df.copy()

    df['HR_signature_SBS3_per_patient'] = df['HR_signature_SBS3_per_patient'].fillna(
        df['HR_signature_SBS3_pretreatment_WGS']
    )

    df['HR_signature_SBS3_per_patient'] = df['HR_signature_SBS3_per_patient'].astype(str).str.strip()
    df['HR_signature_SBS3_per_patient'] = df['HR_signature_SBS3_per_patient'].replace({'nan': 'not_sequenced', 'nd': 'not_sequenced'})
    df['is_hrd'] = np.where(df['HR_signature_SBS3_per_patient'] == 'hrd', 'yes', 'no')

    return df


def clean_chronic_illnesses(val):
    """ Maps illnesses to standard and return a list for every patient. """
    if pd.isna(val):
        return []

    val_str = str(val).strip().lower()

    if val_str in ["no", "nd", "none", "nan", "0", "false"]:
        return []

    illness_patterns = [
        (r"hyperten|htn|high blood pressure", "hypertension"),
        (r"hypercholest|hyperlipid|high cholesterol|dyslipid", "hypercholesterolemia"),
        (r"hypothyr|hypothyreo", "hypothyroidism"),
        (r"hyperthyr|hyperthyreo", "hyperthyroidism"),
        (r"diabet|dm2|dm1|t2d|t1d", "diabetes"),
        (r"astma|asthma", "asthma"),
        (r"copd|ckd|emphysema", "copd"),
        (r"arthr|artrit|arthrosis|artrosi", "arthritis_arthrosis"),
        (r"reflux|gerd|esophag", "gerd"),
        (r"flimmer|afib|atrial|cardiac|heart|coronary|infarct", "cardiovascular_disease"),
        (r"depress|anxi|psych", "psychiatric_condition"),
        (r"epilep", "epilepsy"),
        (r"osteopor", "osteoporosis")
    ]

    found_illnesses = set()
    for pattern, std_name in illness_patterns:
        if re.search(pattern, val_str):
            found_illnesses.add(std_name)

    if not found_illnesses and len(val_str) > 1:
        found_illnesses.add("other_illness")

    return sorted(list(found_illnesses))


def clean_previous_cancers(val):
    """ Maps cancers to standard and return a list for every patient. """
    if pd.isna(val):
        return []

    val_str = str(val).strip().lower()

    if val_str in ["no", "nd", "none", "nan", "0", "false"]:
        return []

    cancer_patterns = [
        (r"breast|mamma|mammary|dcin|ductal", "breast_cancer"),
        (r"lymphoma|lymfoma|hodgkin|myeloma", "hematologic_cancer"),
        (r"skin|basalioma|squamous|scc|melanoma", "skin_cancer"),
        (r"lung", "lung_cancer"),
        (r"colon|rectal|sigma|gastro|ventricle", "gastrointestinal_cancer"),
        (r"endometri|cervic|uter|ovary|borderline", "gynecologic_cancer"),
        (r"renal|kidney|thyroid", "other_organ_cancer")
    ]

    found_cancers = set()
    for pattern, std_name in cancer_patterns:
        if re.search(pattern, val_str):
            found_cancers.add(std_name)

    if not found_cancers and len(val_str) > 1:
        found_cancers.add("other_cancer")

    return sorted(list(found_cancers))

