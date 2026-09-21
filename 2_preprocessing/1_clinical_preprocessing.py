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
    "Age_at_Diagnosis",
    "BMI_at_Dg",

    "Histology",
    # "Histology_other_diagnosis",  # to std categories, too few data
    "Histology_re_evaluated_notes",  # used then deleted

    "Stage_FIGO2014",
    "Treatment_strategy",

    "Residual_tumor_PDS",  #
    "Residual_tumor_IDS",  # --> unified in single feature

    "Oper2_IDS_cancelled",  #
    "Oper2_laparoscopic_evaluation", #
    "Oper2_explorative_laparotomy", #
    "Oper2_debulking_surgery", #  --> unified in single feature

    #"Oper1_Omental_disease_largest_nodule_NEW",  # advanced info
    #"Oper2_Omental_disease_largest_nodule_NEW",  # advanced info
    "Primary_chemotherapy_cycles",
    "NACT_cycles",
    "Post_IDS_chemotherapy_cycles",
    "Maintenance_therary_after_1st_line",
    "PARPi_treatment",
    #"CRS_Omental",
    #"CRS_Omental_range",
    #"RECIST_1.1_Response_to_NACT",
    "Primary_therapy_outcome",
    "Progression_Yes_No",  # target label
    "Survival",  # target label

    "OS_overall_survival_Days_from_Dg_to_death", #
    "OS_current_Time_from_dg_alive__Days", #
    "PFS_Days_from_Dg_to_1st_prog", #
    "PFS_Current_no_prog_alive_days", #
    "PFI_after_Primary_therapy_if_Prog_Days", #
    "PFI_current_no_prog_alive_days", # --> unified in new features

    #"Time_from_End_of_1st_line_maintenance_to_1st_prog_Days_TFI", # just in case of sub-analysis on Maintenance
    # (focus only on patients treated with PARPi / Bevacizumab --> target to evaluate effectiveness

    #"Time_from_1st_prog_to_Death_Days_Post_progression_survival", # just in case target = survival after progression

    "BRCA_mutation_any", #
    "BRCA_mutation_status", #
    "BRCA_any_in_Decider", # --> unified in new feature

    "HR_signature_SBS3_pretreatment_WGS", #
    "HR_signature_SBS3_per_patient", # --> unified in single feature
    "Chronic_illnesses_at_Dg",
    "Chronic_illnesses_type",
    "Previous_cancer_yes_no",
    "Previous_cancer_dg",
]


if __name__ == "__main__":
    target_col = config_file_tree.TARGET_COL

    df_clinical = read_csv(path_to_clinical_file,sep='\t', index_col=0)

    # Filter dataset to available features in file
    available_features = [col for col in features_considered if col in df_clinical.columns]
    df_preprocessed = df_clinical[available_features].copy()

    if 'Histology'in df_preprocessed.columns and 'Histology_re_evaluated_notes' in df_preprocessed.columns:
        df_preprocessed['Histology'] = df_preprocessed.apply(correct_histology, axis=1)
        df_preprocessed = df_preprocessed.drop(columns=['Histology_re_evaluated_notes'])

    if 'Residual_tumor_PDS' in df_preprocessed.columns and 'Residual_tumor_IDS' in df_preprocessed.columns:
        df_preprocessed['Residual_tumor_PDS'] = df_preprocessed['Residual_tumor_PDS'].fillna(df_preprocessed['Residual_tumor_IDS'])
        df_preprocessed = df_preprocessed.rename(columns={'Residual_tumor_PDS': 'Residual_tumor'})
        df_preprocessed = df_preprocessed.drop(columns=['Residual_tumor_IDS'])
        df_preprocessed['Residual_tumor'] = df_preprocessed['Residual_tumor'].apply(clean_residual_tumor)

    cols = [
        'Oper2_IDS_cancelled',
        'Oper2_laparoscopic_evaluation',
        'Oper2_explorative_laparotomy',
        'Oper2_debulking_surgery'
    ]
    df_preprocessed['Oper2_IDS_cancelled'] = df_preprocessed.apply(combine_oper2_cols, axis=1)
    df_preprocessed = df_preprocessed.rename(columns={'Oper2_IDS_cancelled': 'Oper2_outcome'})
    df_preprocessed = df_preprocessed.drop(columns=[c for c in cols if c in df_preprocessed.columns])

    df_preprocessed = clean_and_infer_chemotherapy_features(df_preprocessed)

    if "Maintenance_therary_after_1st_line" in  df_preprocessed.columns:
        df_preprocessed['Maintenance_therary_after_1st_line'] = df_preprocessed['Maintenance_therary_after_1st_line'].apply(
            lambda x: clean_maintenance_therapy(x)
        )

    if 'Primary_therapy_outcome' in df_preprocessed.columns:
        df_preprocessed['Primary_therapy_outcome'] = df_preprocessed['Primary_therapy_outcome'].apply(map_primary_outcome_binary)

    if 'Survival' in df_preprocessed.columns:
        df_preprocessed['Survival'] = df_preprocessed['Survival'].apply(clean_survival)

    cols = [
        'OS_overall_survival_Days_from_Dg_to_death',
        'OS_current_Time_from_dg_alive__Days',
        'PFS_Days_from_Dg_to_1st_prog',
        'PFS_Current_no_prog_alive_days',
        'PFI_after_Primary_therapy_if_Prog_Days',
        'PFI_current_no_prog_alive_days',
    ]
    if any(col not in df_clinical.columns for col in cols):
        raise KeyError("CHECK some of these columns are missing: " + str(cols))

    df_preprocessed = process_survival_time_features(df_preprocessed)
    df_preprocessed = df_preprocessed.drop(columns=cols)


    df_preprocessed['BRCA_mutation_any'] = df_preprocessed.apply(clean_brca_mutation, axis=1)
    df_preprocessed = df_preprocessed.drop(columns=['BRCA_any_in_Decider', 'BRCA_mutation_status'])

    if 'HR_signature_SBS3_per_patient' and 'HR_signature_SBS3_pretreatment_WGS' in df_preprocessed.columns:
        df_preprocessed = clean_hrd_status(df_preprocessed)
        df_preprocessed = df_preprocessed.drop(columns=['HR_signature_SBS3_pretreatment_WGS'])

    if 'Chronic_illnesses_type' in df_preprocessed.columns:
        df_preprocessed['Chronic_illnesses_type'] = df_preprocessed['Chronic_illnesses_type'].apply(clean_chronic_illnesses)

    if 'Previous_cancer_dg' in df_preprocessed.columns:
        df_preprocessed['Previous_cancer_dg'] = df_preprocessed['Previous_cancer_dg'].apply(clean_previous_cancers)

    print("Excluding samples where target = nan...")
    df_preprocessed = df_preprocessed.dropna(subset=[target_col]).copy()
    if config_file_tree.TASK == 'survival':
        time_col = config_file_tree.TIME_COL
        df_preprocessed = df_preprocessed[df_preprocessed[time_col] > 0]  # remove non-valid times
        df_preprocessed = df_preprocessed.dropna(subset=[time_col])

    # Save preprocessed clinical data
    df_preprocessed.to_csv(output_filepath, sep='\t', index=False)

    print(f"Final columns ({len(df_preprocessed.columns)}): {df_preprocessed.columns.values}")

    print("\nDONE!\n")
