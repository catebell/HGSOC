from os import path
from pandas import read_csv

import config_file_tree
from utils_clinical_clean_methods import *

extracted_dir = config_file_tree.EXTRACTED_DATA_DIR
path_to_clinical_file = path.join(extracted_dir, "raw_clinical_data.tsv")

output_dir = config_file_tree.PREPROCESSED_DATA_DIR
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
    "Post_IDS_chemotherapy_cycles",

    "Primary_therapy_outcome",
    "Progression_Yes_No",  # target label
    "Survival",  # target label
    "Cause_of_death",

    "OS_KaplanM_allHGSC",
    "PFS_KaplanM_allHGSC",
    "PFI_KaplanM_allHGSC",

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

# TODO
    '''
    #"Oper1_Omental_disease_largest_nodule_NEW",  # advanced info
    #"Oper2_Omental_disease_largest_nodule_NEW",  # advanced info
    
    "Maintenance_therary_after_1st_line",
    "PARPi_treatment",
    #"CRS_Omental",
    #"CRS_Omental_range",
    #"RECIST_1.1_Response_to_NACT",
    

    #"Time_from_End_of_1st_line_maintenance_to_1st_prog_Days_TFI", # just in case of sub-analysis on Maintenance
    # (focus only on patients treated with PARPi / Bevacizumab --> target to evaluate effectiveness

    #"Time_from_1st_prog_to_Death_Days_Post_progression_survival", # just in case target = survival after progression
    
    '''
]


if __name__ == "__main__":
    target_col = config_file_tree.TARGET_COL

    df_clinical = read_csv(path_to_clinical_file,sep='\t', index_col=0)

    # Filter dataset to available features in file
    available_features = [col for col in features_considered if col in df_clinical.columns]
    df_clinical = df_clinical[available_features].copy()

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
    cols = [
        'OS_overall_survival_Days_from_Dg_to_death',
        'OS_current_Time_from_dg_alive__Days',
        'PFS_Days_from_Dg_to_1st_prog',
        'PFS_Current_no_prog_alive_days',
        'PFI_after_Primary_therapy_if_Prog_Days',
        'PFI_current_no_prog_alive_days',
    ]
    if all(col in df_clinical.columns for col in cols):
        df_clinical = process_survival_time_features(df_clinical)
        df_clinical = df_clinical.drop(columns=cols)
    '''

    if 'PFI_KaplanM_allHGSC' in df_clinical.columns:
        df_clinical['PFI_target'] = df_clinical['PFI_KaplanM_allHGSC'].apply(assign_pfi_class)

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

    print("Excluding samples where target = nan...")
    if target_col in df_clinical.columns:
        df_clinical = df_clinical.dropna(subset=[target_col]).copy()
        if config_file_tree.TASK == 'survival':
            time_col = config_file_tree.TIME_COL
            df_clinical = df_clinical[df_clinical[time_col] > 0]  # remove non-valid times
            df_clinical = df_clinical.dropna(subset=[time_col])

    # Save preprocessed clinical data
    df_clinical.to_csv(output_filepath, sep='\t', index=False)

    print(f"Final columns ({len(df_clinical.columns)}): {df_clinical.columns.values}")

    print("\nDONE!\n")
