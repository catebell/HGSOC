from os import path, makedirs
from pandas import read_csv

import config_file_tree
from utils_clinical_clean_methods import *

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
path_to_clinical_file = path.join(extracted_dir, "raw_clinical_data.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
makedirs(output_dir, exist_ok=True)

output_filepath = path.join(output_dir, "preprocessed_clinical_data.tsv")

features_considered = [
    "Patient_cohort_code_Patient_Card",
    "Age_at_Diagnosis",  # feature
    "BMI_at_Dg",  # feature

    "Histology",  # integrated with Histology_re_evaluated_notes and used only for filtering
    "Histology_re_evaluated_notes",  # used then deleted

    "Stage_FIGO2014",  # feature or label, for now not considered
    "Treatment_strategy",  # auxiliary

    "Residual_tumor_PDS",  #
    "Residual_tumor_IDS",  # --> unified in Residual_Tumor

    "Oper2_IDS_cancelled",  #
    "Oper2_laparoscopic_evaluation", #
    "Oper2_explorative_laparotomy", #
    "Oper2_debulking_surgery", #  --> unified in Oper2_outcome

    "Primary_chemotherapy_cycles",
    "NACT_cycles",
    "RECIST_1.1_Response_to_NACT",
    "Post_IDS_chemotherapy_cycles",

    "Primary_therapy_outcome",
    "Progression_Yes_No",  # target label
    "Survival",  # target label
    "Cause_of_death",

    "OS_KaplanM_allHGSC",
    "PFS_KaplanM_allHGSC",

    "PFI_KaplanM_allHGSC", #
    "Excluded_from_PFI_calculations", #
    "Excluded_from_PFI_calculations__reason", # --> unified in PFI_KaplanM_allHGSC + flag is_platinum_resistant

    # old names
    #"OS_overall_survival_Days_from_Dg_to_death", #
    #"OS_current_Time_from_dg_alive__Days", # --> unified in OS_days

    #"PFS_Days_from_Dg_to_1st_prog", #
    #"PFS_Current_no_prog_alive_days", # --> unified in PFS_days

    #"PFI_after_Primary_therapy_if_Prog_Days", #
    #"PFI_current_no_prog_alive_days", # --> unified in PFI_days + flag is_platinum_resistant

    "HR_signature_SBS3_pretreatment_WGS", #
    "HR_signature_SBS3_per_patient", # --> unified in HR_signature_SBS3_per_patient + flag is_hdr

    "BRCA_mutation_any", #
    "BRCA_mutation_status_in_Clinical_test", #
    "BRCA_any_in_Decider", # --> unified in BRCA_mutation_any

    "Chronic_illnesses_at_Dg",
    "Chronic_illnesses_type",
    "Previous_cancer_yes_no",
    "Previous_cancer_dg",

    "Progression_date_ND__No_prog_at_regular_follow_up_but_no_further_follow_up",
]


if __name__ == "__main__":
    df_clinical = read_csv(path_to_clinical_file,sep='\t', index_col=0)

    # Filter dataset to available features in file
    available_features = [col for col in features_considered if col in df_clinical.columns]
    df_clinical = df_clinical[available_features].copy()

    if 'Patient_cohort_code_Patient_Card' in df_clinical.columns:
        df_clinical = df_clinical.rename(columns={'Patient_cohort_code_Patient_Card': 'patient_id'})
        df_clinical['patient_id'] = df_clinical['patient_id'].str.upper()

    if 'Histology'in df_clinical.columns:
        if 'Histology_re_evaluated_notes' in df_clinical.columns:
            df_clinical['Histology'] = df_clinical.apply(correct_histology, axis=1)
            df_clinical = df_clinical.drop(columns=['Histology_re_evaluated_notes'])

    if 'Residual_tumor_PDS' in df_clinical.columns and 'Residual_tumor_IDS' in df_clinical.columns:
        df_clinical['Residual_tumor_PDS'] = df_clinical['Residual_tumor_PDS'].fillna(df_clinical['Residual_tumor_IDS'])
        df_clinical = df_clinical.rename(columns={'Residual_tumor_PDS': 'Residual_tumor'})
        df_clinical = df_clinical.drop(columns=['Residual_tumor_IDS'])
        df_clinical['Residual_tumor'] = df_clinical['Residual_tumor'].apply(clean_residual_tumor)

    cols = [
        'Oper2_IDS_cancelled',
        'Oper2_laparoscopic_evaluation',
        'Oper2_explorative_laparotomy',
        'Oper2_debulking_surgery'
    ]
    df_clinical['Oper2_IDS_cancelled'] = df_clinical.apply(combine_oper2_cols, axis=1)
    df_clinical = df_clinical.rename(columns={'Oper2_IDS_cancelled': 'Oper2_outcome'})
    df_clinical = df_clinical.drop(columns=[c for c in cols if c in df_clinical.columns])

    df_clinical = clean_and_infer_chemotherapy_features(df_clinical)

    if "Maintenance_therary_after_1st_line" in  df_clinical.columns:
        df_clinical['Maintenance_therary_after_1st_line'] = df_clinical['Maintenance_therary_after_1st_line'].apply(
            lambda x: clean_maintenance_therapy(x)
        )

    if 'Primary_therapy_outcome' in df_clinical.columns:
        df_clinical['Primary_therapy_outcome'] = df_clinical['Primary_therapy_outcome'].apply(map_primary_outcome_binary)

    if 'Cause_of_death' in df_clinical.columns:
        df_clinical['Cause_of_death'] = df_clinical['Cause_of_death'].fillna(df_clinical['Survival'])
        df_clinical['Survival'] = df_clinical['Cause_of_death'].apply(clean_survival)
        df_clinical = df_clinical.drop(columns=['Cause_of_death'])

    ''' old names
    cols_order = [
        'OS_overall_survival_Days_from_Dg_to_death',
        'OS_current_Time_from_dg_alive__Days',
        'PFS_Days_from_Dg_to_1st_prog',
        'PFS_Current_no_prog_alive_days',
        'PFI_after_Primary_therapy_if_Prog_Days',
        'PFI_current_no_prog_alive_days',
    ]
    if all(col in df_clinical.columns for col in cols_order):
        df_clinical = process_survival_time_features(df_clinical)
        df_clinical = df_clinical.drop(columns=cols_order)
    '''

    cols = ['PFI_KaplanM_allHGSC', 'Excluded_from_PFI_calculations', 'Excluded_from_PFI_calculations__reason']
    if 'PFI_KaplanM_allHGSC' in df_clinical.columns:
        df_clinical = process_pfi_features(df_clinical)
        df_clinical['Excluded_from_PFI_calculations'] = df_clinical['Excluded_from_PFI_calculations'].fillna('no')
        for col in cols:
            if col in df_clinical.columns and col not in ['PFI_KaplanM_allHGSC', 'Excluded_from_PFI_calculations']:
                df_clinical = df_clinical.drop(columns=[col])


    if all(col in df_clinical.columns for col in ['BRCA_mutation_any', 'BRCA_any_in_Decider', 'BRCA_mutation_status_in_Clinical_test']):
        df_clinical['BRCA_mutation_any'] = df_clinical.apply(clean_brca_mutation, axis=1)
        df_clinical = df_clinical.drop(columns=['BRCA_any_in_Decider', 'BRCA_mutation_status_in_Clinical_test'])

    if 'HR_signature_SBS3_per_patient' and 'HR_signature_SBS3_pretreatment_WGS' in df_clinical.columns:
        df_clinical = clean_hrd_status(df_clinical)
        df_clinical = df_clinical.drop(columns=['HR_signature_SBS3_pretreatment_WGS'])

    if 'Chronic_illnesses_type' in df_clinical.columns:
        df_clinical['Chronic_illnesses_type'] = df_clinical['Chronic_illnesses_type'].apply(clean_chronic_illnesses)

    if 'Previous_cancer_dg' in df_clinical.columns:
        df_clinical['Previous_cancer_dg'] = df_clinical['Previous_cancer_dg'].apply(clean_previous_cancers)

    if 'Progression_date_ND__No_prog_at_regular_follow_up_but_no_further_follow_up' in  df_clinical.columns:
        df_clinical['Progression_date_ND__No_prog_at_regular_follow_up_but_no_further_follow_up'] = df_clinical['Progression_date_ND__No_prog_at_regular_follow_up_but_no_further_follow_up'].fillna('no')

    # Chemo-refractory vs Chemo-sensitive target definition
    cond_refractory = (
        df_clinical['RECIST_1.1_Response_to_NACT'].isin(
            ['stable disease after nact', 'progressive disease after nact']
        ) & (df_clinical['PFI_KaplanM_allHGSC'] <= 45.0)
    )

    cond_sensitive = (
        df_clinical['RECIST_1.1_Response_to_NACT'].isin(
            ['partial response to nact', 'complete response to nact']
        )& (
            df_clinical['PFI_KaplanM_allHGSC'].isna() | (df_clinical['PFI_KaplanM_allHGSC'] > 180.0)
        )
    )

    conditions = [cond_refractory, cond_sensitive]
    choices = [1, 0]
    df_clinical['chemo_refractory_target'] = np.select(conditions, choices, default=np.nan)

    df_clinical = df_clinical.drop(columns=['RECIST_1.1_Response_to_NACT'])

    # Save preprocessed clinical data
    df_clinical.to_csv(output_filepath, sep='\t', index=False)

    print(f"Final columns ({len(df_clinical.columns)}): {df_clinical.columns.tolist()}")

    print("\nDONE!\n")
