import pandas as pd
import numpy as np
import re
from psutil import Process
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, MultiLabelBinarizer, OneHotEncoder

import config_file_tree

""""" utility """""

def print_details(dataset):
    print()
    print(f'Train Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')

    print(dataset[0])
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {dataset[0].num_nodes}')
    print(f'Number of edges: {dataset[0].num_edges}')
    print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
    print(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
    # print(f'Has self-loops: {dataset[0].has_self_loops()}')  # added internally during GNN forward, checked in gatv2_conv module
    print(f'Is undirected: {dataset[0].is_undirected()}')
    print()


def get_memory_usage():
    process: Process = Process(getpid())
    megabytes: float = process.memory_info().rss / 1024 ** 2
    print(f"Used {megabytes:.2f} MB")


""""" clinical pre training preprocessing """""

# filter cols to prevent data leakage
LEAKAGE_MAP = { # todo completare elenco rimozione colonne di leakage o superflue per la task
    # Primary Therapy Response Yes No classification:

    # Platinum resistant classification:
    'platinum_resistant': ['PFI_days', 'PFS_days', 'Progression_Yes_No', 'had_progression_therapy'],  # todo check

    # Overall Survival task:
    'Survival': ['PFI_days', 'platinum_resistant'],  # todo check

    # Progression Free Survival:
    'PFS_days': ['had_progression_therapy', 'PFI_days', 'platinum_resistant', 'Survival', 'OS_days', 'Primary_therapy_outcome'],

    # Progression Yes No classification:
    'Progression_Yes_No': ['PFS_days', 'PFI_days', 'platinum_resistant', 'had_progression_therapy']  # todo check
}


def pre_train_imputation_encoding(df, task, target_col = None, is_train=True, encoders=None, scalers=None, imputers=None, categories=None):
    """
    Clinical nan imputation, reformatting, normalization and target column codification preventing data leakage.
    If is_train=True: compute parameters on Train data, else applies same parameters on Test/Validation data too.
    """
    df_norm = df.copy()
    target_col_encoded = False

    if is_train:
        encoders = {}
        scalers = {}
        imputers = {}
    else:
        if encoders is None or scalers is None or imputers is None:
            raise ValueError("Encoders, Scalers and Imputers are needed when is_train=False!")

    if target_col not in df_norm.columns:
        raise KeyError(f"CHECK target column ({target_col}) is missing from data! ")

    leakage_cols = LEAKAGE_MAP.get(target_col, []) if target_col else []
    cols_to_drop = set(leakage_cols)

    if cols_to_drop:
        df_norm = df_norm.drop(columns=[col for col in cols_to_drop if col in df_norm.columns])

    if task == 'classification':
        target_cols = [target_col]
    elif task == 'survival':
        target_cols = [target_col, config_file_tree.TIME_COL]

    features_cols = [col for col in df_norm.columns if col not in target_cols]


    """"" Numerical Cols : imputer (median) for missing data + std scaler """""

    cols = ['Age_at_Diagnosis', 'BMI_at_Dg']
    for col in cols:
        if col in features_cols:
            if is_train:
                imputers[col] = SimpleImputer(strategy='median')
                scalers[col] = StandardScaler()

                imputed_vals = imputers[col].fit_transform(df_norm[[col]])
                df_norm[col] = scalers[col].fit_transform(imputed_vals).ravel()
            else:
                imputed_vals = imputers[col].transform(df_norm[[col]])
                df_norm[col] = scalers[col].transform(imputed_vals).ravel()


    """"" DAYS features : Log1p transform + std scaler """""

    time_cols = [col for col in features_cols if 'days' in col.lower()]
    for col in time_cols:
        if is_train:
            imputers[col] = SimpleImputer(strategy='median')
            scalers[col] = StandardScaler()

            imputed_vals = imputers[col].fit_transform(df_norm[[col]])
            log_vals = np.log1p(np.maximum(0, imputed_vals))
            df_norm[col] = scalers[col].fit_transform(log_vals).ravel()
        else:
            imputed_vals = imputers[col].transform(df_norm[[col]])
            log_vals = np.log1p(np.maximum(0, imputed_vals))
            df_norm[col] = scalers[col].transform(log_vals).ravel()


    """"" THERAPY CYCLES : RobustScaler (outliers resistant) """""

    cols_cycles = [col for col in features_cols if 'cycles' in col.lower()]
    for col in cols_cycles:
        if is_train:
            imputers[col] = SimpleImputer(strategy='median')
            scalers[col] = RobustScaler()

            imputed_vals = imputers[col].fit_transform(df_norm[[col]])
            df_norm[col] = scalers[col].fit_transform(imputed_vals).ravel()
        else:
            imputed_vals = imputers[col].transform(df_norm[[col]])
            df_norm[col] = scalers[col].transform(imputed_vals).ravel()


    """"" ORDINAL FEATURES MinMaxScaler() """""

    ordinal_mappings = {
        'Stage_FIGO2014': ['i', 'ia', 'ib', 'ic', 'ii', 'iia', 'iib', 'iic', 'iii', 'iiia', 'iiib', 'iiic', 'iv', 'iva',
                           'ivb'],
        'Residual_tumor': ['0', '1-10mm', '>10mm'],
        'Oper1_Omental_disease_largest_nodule_NEW': ['no', 'under 2cm', '2-5cm', 'over 5cm'],
        'Oper2_Omental_disease_largest_nodule_NEW': ['no', 'under 2cm', '2-5cm', 'over 5cm'],
    }

    cols_ordinal = [col for col in ordinal_mappings.keys() if col in features_cols]

    for col in cols_ordinal:
        df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
        ordinal_categories = ordinal_mappings[col]
        if is_train:
            encoders[col] = OrdinalEncoder(
                categories=[ordinal_categories],
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
            imputers[col] = SimpleImputer(strategy='median')
            scalers[col] = MinMaxScaler()

            encoded_vals = encoders[col].fit_transform(df_norm[[col]])
            imputed_vals = imputers[col].fit_transform(encoded_vals)
            df_norm[col] = scalers[col].fit_transform(imputed_vals).ravel()
        else:
            encoded_vals = encoders[col].transform(df_norm[[col]])
            imputed_vals = imputers[col].transform(encoded_vals)
            df_norm[col] = scalers[col].transform(imputed_vals).ravel()


    """"" BOOLEAN FEATURES 'most frequent' strategy """""

    boolean_mappings = {
        'PARPi_treatment': ['no', 'yes'],
        'Progression_Yes_No': ['no', 'yes'],
        'BRCA_mutation_any': ['no', 'yes'],
        'Chronic_illnesses_at_Dg': ['no', 'yes'],
        'Previous_cancer_yes_no': ['no', 'yes'],
        'is_platinum_resistant': ['no', 'yes'],
        'is_hrd': ['no', 'yes'],
        'is_fluid': ['no', 'yes'],
    }

    cols_boolean = [col for col in boolean_mappings.keys() if col in features_cols or col == target_col]

    for col in cols_boolean:
        df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
        bool_categories = boolean_mappings[col]
        if is_train:
            encoders[col] = OrdinalEncoder(
                categories=[bool_categories],
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
            imputers[col] = SimpleImputer(strategy='most_frequent')  # target nans already filtered out in preprocessing

            encoded_vals = encoders[col].fit_transform(df_norm[[col]])
            df_norm[col] = imputers[col].fit_transform(encoded_vals).ravel()
            if col == target_col: target_col_encoded = True
        else:
            encoded_vals = encoders[col].transform(df_norm[[col]])
            df_norm[col] = imputers[col].transform(encoded_vals).ravel()


    """"" BOOLEAN SURVIVAL FEATURES 'censor/fill 0' strategy """""

    bool_censor_mappings = {
        'Survival': ['alive', 'dead'],
        'Primary_therapy_outcome': ['unresponsive', 'responsive']
    }

    cols_boolean = [col for col in bool_censor_mappings.keys() if col in features_cols or col == target_col]

    for col in cols_boolean:
        df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
        bool_categories = bool_censor_mappings[col]
        if is_train:
            encoders[col] = OrdinalEncoder(
                categories=[bool_categories],
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
            imputers[col] = SimpleImputer(strategy='constant', fill_value=0)  # target nans already filtered out in preprocessing

            encoded_vals = encoders[col].fit_transform(df_norm[[col]])
            df_norm[col] = imputers[col].fit_transform(encoded_vals).ravel()
            if col == target_col: target_col_encoded = True
        else:
            encoded_vals = encoders[col].transform(df_norm[[col]])
            df_norm[col] = imputers[col].transform(encoded_vals).ravel()


    """"" MULTI-LABEL CATEGORICAL FEATURES MultiLabelBinarizer strategy for values = lists """""

    cols_for_mlb = ['Maintenance_therary_after_1st_line', 'Chronic_illnesses_type', 'Previous_cancer_dg']

    for col in cols_for_mlb:
        if col in features_cols:
            df_norm[col] = df_norm[col].astype(str).apply(
                lambda x: re.findall(r"[a-zA-Z0-9_]+", x)
            )  # reads as list

            if is_train:
                encoders[col] = MultiLabelBinarizer()
                multi_hot_matrix = encoders[col].fit_transform(df_norm[col])
            else:
                multi_hot_matrix = encoders[col].transform(df_norm[col])

            # new df with cols "col_Value"
            new_col_names = [f"{col}_{str(cls).replace(' ', '_')}" for cls in encoders[col].classes_]
            new_df = pd.DataFrame(
                multi_hot_matrix,
                columns=new_col_names,
                index=df_norm.index
            )
            df_norm = (pd.concat([df_norm, new_df], axis=1))
            df_norm = df_norm.drop(columns=col)


    """"" CATEGORICAL FEATURES ENCODING One-Hot Encoding strategy for mutually exclusive features """""

    cols_for_ohe = ['Histology', 'Treatment_strategy', 'Oper2_outcome', 'HR_signature_SBS3_per_patient', 'phase_label', 'tissue']
    existing_ohe_cols = [col for col in cols_for_ohe if col in features_cols]

    if existing_ohe_cols:
        df_norm[existing_ohe_cols] = df_norm[existing_ohe_cols].fillna('missing').astype(str)

        if is_train:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
            encoders['one_hot'] = ohe
            ohe_matrix = ohe.fit_transform(df_norm[existing_ohe_cols])
        else:
            ohe = encoders['one_hot']
            # unseen categories will have all 0s
            ohe_matrix = ohe.transform(df_norm[existing_ohe_cols])

        feature_names = ohe.get_feature_names_out(existing_ohe_cols)
        ohe_df = pd.DataFrame(ohe_matrix, columns=feature_names, index=df_norm.index)

        df_norm = pd.concat([df_norm, ohe_df], axis=1)
        df_norm = df_norm.drop(columns=existing_ohe_cols)


    """"" target codification into zero-based continuous interval for pytorch functions (es. [6,7,9] -> [0,1,3]) """""

    if not target_col_encoded:  # TODO finire
        # processing with ordinal encoder o label encoder
        print()

    '''
    if task == 'classification':
        if is_train:
            categories = sorted(df_norm[target_col].dropna().unique())
        else:
            if categories is None:
                raise ValueError("categories are needed when task = 'classification' and is_train=False.")

        df_norm[target_col] = pd.Categorical(df_norm[target_col], categories=categories).codes
    '''

    if is_train:
        return df_norm, encoders, scalers, imputers, categories
    return df_norm


if __name__ == '__main__':
    from os import path, getpid

    task = config_file_tree.TASK
    target_col = config_file_tree.TARGET_COL

    df_samples_metadata = pd.read_csv(path.join('2_preprocessing', 'preprocessed_data', 'data_HERCULES_DECIDER', 'preprocessed_samples_metadata.tsv'), sep='\t')
    #df_clinical = pd.read_csv(path.join('2_preprocessing', 'preprocessed_data', 'data_HERCULES_DECIDER', 'preprocessed_clinical_data.tsv'), sep='\t')
    df_new, enc, scal, imp, cat = pre_train_imputation_encoding(df_samples_metadata, task, target_col, is_train=True,)

    print("breakpoint")