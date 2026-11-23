import json
import logging
import os
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from psutil import Process
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

import config_file_tree
from utils_explainability import to_rel_path

""""" utility """""

def print_details(dataset):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('last_execution.log', mode='w'),
            logging.StreamHandler()
        ]
    )

    print()
    logging.info(f'Train Dataset: {dataset}:')
    logging.info('====================')
    logging.info(f'Number of graphs: {len(dataset)}')
    logging.info(f"Node features ({dataset.num_features}): {getattr(dataset[0], 'node_feature_names', 'Unspecified')}")
    logging.info(f"Node tensor shape: {dataset[0].x.shape}")

    logging.info(dataset[0])
    logging.info('=============================================================')

    # Gather some statistics about the first graph.
    logging.info(f'Number of nodes: {dataset[0].num_nodes}')
    logging.info(f'Number of edges: {dataset[0].num_edges}')
    logging.info(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
    logging.info(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
    # print(f'Has self-loops: {dataset[0].has_self_loops()}')  # added internally during GNN forward, checked in gatv2_conv module
    logging.info(f'Is undirected: {dataset[0].is_undirected()}')
    print()

    return dataset[0].num_nodes


def get_memory_usage():
    process: Process = Process(os.getpid())
    megabytes: float = process.memory_info().rss / 1024 ** 2
    print(f"Used {megabytes:.2f} MB")


""""" clinical pre training preprocessing """""

def pre_train_imputation_encoding(df, task, target_col = None, is_train=True, encoders=None, scalers=None, imputers=None, categories=None):
    """
    Clinical nan imputation, reformatting, normalization and target column codification preventing data leakage.
    If is_train=True: compute parameters on Train data, else applies same parameters on Test/Validation data too.
    """
    df_norm = df.copy()

    if is_train:
        encoders = {}
        scalers = {}
        imputers = {}
    else:
        if encoders is None or scalers is None or imputers is None:
            raise ValueError("Encoders, Scalers and Imputers are needed when is_train=False!")

    if target_col not in df_norm.columns:
        raise KeyError(f"CHECK target column ({target_col}) is missing from data! ")

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

    time_cols = [col for col in features_cols if 'kaplan' in col.lower() or 'days' in col.lower() or 'time' in col.lower()]
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

    cols_boolean = [col for col in boolean_mappings.keys() if col in features_cols]
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
        else:
            encoded_vals = encoders[col].transform(df_norm[[col]])
            df_norm[col] = imputers[col].transform(encoded_vals).ravel()


    """"" BOOLEAN SURVIVAL FEATURES 'censor/fill 0' strategy """""

    bool_censor_mappings = {
        'Survival': ['alive', 'dead'],
        'Primary_therapy_outcome': ['unresponsive', 'responsive']
    }

    cols_boolean = [col for col in bool_censor_mappings.keys() if col in features_cols]
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

            # new df with cols_to_keep "col_Value"
            new_col_names = [f"{col}_{str(cls).replace(' ', '_')}" for cls in encoders[col].classes_]
            new_df = pd.DataFrame(
                multi_hot_matrix,
                columns=new_col_names,
                index=df_norm.index
            )
            df_norm = (pd.concat([df_norm, new_df], axis=1))
            df_norm = df_norm.drop(columns=col)


    """"" CATEGORICAL FEATURES ENCODING One-Hot Encoding strategy for mutually exclusive features """""

    cols_for_ohe = ['Histology', 'Treatment_strategy', 'Oper2_outcome', 'HR_signature_SBS3_per_patient', 'phase_label', 'tissue', 'tissue_type']
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

    if is_train:
        return df_norm, encoders, scalers, imputers, categories
    return df_norm


def _build_training_config_dict():
    """Build a JSON-serializable snapshot of config_file_tree parameters."""
    active_idx = list(getattr(config_file_tree, "NODE_FEATURE_INDEX", []))
    all_names = list(getattr(config_file_tree, "NODE_FEATURE_NAMES", []))
    task = getattr(config_file_tree, "TASK", "N/A")
    use_meta = bool(getattr(config_file_tree, "USE_SAMPLES_METADATA", False))
    use_clin = bool(getattr(config_file_tree, "USE_CLINICAL_DATA", False))
    split_dir = getattr(config_file_tree, "SPLIT_DIR", "")
    graphs_dir = getattr(config_file_tree, "PROCESSED_GRAPHS_DIR", "")
    samples_dir = getattr(config_file_tree, "PROCESSED_SAMPLES_DIR", "")

    return {
        "DATASET_NAME": getattr(config_file_tree, "DATASET_NAME", "N/A"),
        "TASK": task,
        "GRAPH_EDGES_TYPE": getattr(config_file_tree, "GRAPH_EDGES_TYPE", "N/A"),
        "SAMPLES_LEVEL_TRAIN": bool(getattr(config_file_tree, "SAMPLES_LEVEL_TRAIN", False)),
        "PATIENTS_LEVEL_TRAIN": bool(getattr(config_file_tree, "PATIENTS_LEVEL_TRAIN", False)),
        "FREEZE_GAT_AFTER_EPOCH": getattr(config_file_tree, "FREEZE_GAT_AFTER_EPOCH", None),
        "RUN_PERMUTATION_TEST": bool(getattr(config_file_tree, "RUN_PERMUTATION_TEST", False)),
        "SAMPLE_TARGET_COL": getattr(config_file_tree, "SAMPLE_TARGET_COL", "N/A"),
        "PATIENT_TARGET_COL": getattr(config_file_tree, "PATIENT_TARGET_COL", "N/A"),
        "TIME_COL": getattr(config_file_tree, "TIME_COL", "N/A") if task == "survival" else None,
        "NODE_FEATURE_INDEX": active_idx,
        "NODE_FEATURE_NAMES": all_names,
        "ACTIVE_NODE_FEATURES": [all_names[i] for i in active_idx if i < len(all_names)],
        "USE_SAMPLES_METADATA": use_meta,
        "METADATA_FEATURES": list(getattr(config_file_tree, "METADATA_FEATURES", [])) if use_meta else [],
        "USE_CLINICAL_DATA": use_clin,
        "CLINICAL_FEATURES": list(getattr(config_file_tree, "CLINICAL_FEATURES", [])) if use_clin else [],
        "SAMPLE_ENCODER_FOLDER": getattr(config_file_tree, "SAMPLE_ENCODER_FOLDER", "N/A"),
        "PATIENT_ENCODER_FOLDER": getattr(config_file_tree, "PATIENT_ENCODER_FOLDER", "N/A"),
        "SPLIT_DIR_REL": to_rel_path(split_dir) if split_dir else "",
        "PROCESSED_GRAPHS_DIR_REL": to_rel_path(graphs_dir) if graphs_dir else "",
        "PROCESSED_SAMPLES_DIR_REL": to_rel_path(samples_dir) if samples_dir else "",
    }


def save_training_config_report(output_dir, patient_subdir="MIL"):
    """
    Save 'training_config.txt' and 'training_config.json' inside active model subdirectories
    ('GAT' when SAMPLES_LEVEL_TRAIN=True, and `patient_subdir` when PATIENTS_LEVEL_TRAIN=True
    and `patient_subdir` is non-empty).
    """
    cfg_dict = _build_training_config_dict()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg_dict["TIMESTAMP"] = timestamp_str

    config_text = f"""================================================================================
TRAINING RUN CONFIGURATION REPORT
Timestamp: {timestamp_str}
================================================================================

[GENERAL CONFIGURATION]
Dataset Name:               {cfg_dict['DATASET_NAME']}
Task Type:                  {cfg_dict['TASK']}
Graph Edges Topology:       {cfg_dict['GRAPH_EDGES_TYPE']}

[TRAINING PHASES & PIPELINE CONTROL]
Samples Level Train (GAT):  {cfg_dict['SAMPLES_LEVEL_TRAIN']}
Patients Level Train (MIL): {cfg_dict['PATIENTS_LEVEL_TRAIN']}
Freeze GAT After Epoch:     {cfg_dict['FREEZE_GAT_AFTER_EPOCH']}
Run Permutation Test:       {cfg_dict['RUN_PERMUTATION_TEST']}

[TARGET LABELS]
Sample Target Label:        {cfg_dict['SAMPLE_TARGET_COL']}
Patient Target Label:       {cfg_dict['PATIENT_TARGET_COL'] if cfg_dict['PATIENTS_LEVEL_TRAIN'] else 'N/A (Patient Level Disabled)'}
Time Target Label (Cox):    {cfg_dict['TIME_COL'] if cfg_dict['TASK'] == 'survival' else 'N/A'}

[GRAPH NODE FEATURES]
Active Node Feature Indices: {cfg_dict['NODE_FEATURE_INDEX']}
Active Node Feature Names:   {cfg_dict['ACTIVE_NODE_FEATURES']}

[METADATA & CLINICAL FEATURES]
Use Samples Metadata:       {cfg_dict['USE_SAMPLES_METADATA']}
Metadata Features Used:     {cfg_dict['METADATA_FEATURES'] if cfg_dict['USE_SAMPLES_METADATA'] else 'None'}
Use Patient Clinical Data:  {cfg_dict['USE_CLINICAL_DATA']}
Clinical Features Used:     {cfg_dict['CLINICAL_FEATURES'] if cfg_dict['USE_CLINICAL_DATA'] else 'None'}

[PRETRAINED CHECKPOINT PATHS]
Pretrained Sample Encoder Folder (GAT):  {cfg_dict['SAMPLE_ENCODER_FOLDER']}
Pretrained Patient Encoder Folder (MIL): {cfg_dict['PATIENT_ENCODER_FOLDER']}
================================================================================
"""

    target_dirs = []
    if cfg_dict["SAMPLES_LEVEL_TRAIN"]:
        target_dirs.append(os.path.join(output_dir, "GAT"))
    if cfg_dict["PATIENTS_LEVEL_TRAIN"] and patient_subdir:
        target_dirs.append(os.path.join(output_dir, patient_subdir))

    saved_paths = []
    for t_dir in target_dirs:
        os.makedirs(t_dir, exist_ok=True)
        txt_path = os.path.join(t_dir, "training_config.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(config_text)
        with open(os.path.join(t_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=4)
        saved_paths.append(txt_path)
        logging.info(f"Training configuration report (TXT + JSON) saved to: {t_dir}")

    return (saved_paths[0] if saved_paths else None), config_text


def handle_pretrained_gat_checkpoint(gat_pretrained_dir, gat_output_dir, config_text):
    """
    Clone a pre-trained GAT checkpoint folder into the run's GAT/ directory and write
    a provenance notice ('pretrained_checkpoint_info.txt').
    """
    if not os.path.exists(gat_pretrained_dir):
        logging.warning(f"Pre-trained GAT folder not found at: {gat_pretrained_dir}")
        return

    abs_src, abs_dst = os.path.abspath(gat_pretrained_dir), os.path.abspath(gat_output_dir)
    if abs_src == abs_dst or (os.path.exists(gat_output_dir) and os.path.samefile(gat_pretrained_dir, gat_output_dir)):
        logging.info(f"Destination GAT folder matches pre-trained folder ({gat_output_dir}). Skipping cloning.")
        return

    os.makedirs(gat_output_dir, exist_ok=True)
    logging.info(f"Cloning pre-trained GAT checkpoint: '{gat_pretrained_dir}' -> '{gat_output_dir}'...")

    for item in os.listdir(gat_pretrained_dir):
        if item == "explained":
            continue
        src_item, dst_item = os.path.join(gat_pretrained_dir, item), os.path.join(gat_output_dir, item)
        if os.path.exists(dst_item) and os.path.samefile(src_item, dst_item):
            continue
        if os.path.isdir(src_item):
            if os.path.exists(dst_item):
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

    # Legacy fallback if source GAT stored training_config.txt in its parent folder
    dst_txt, dst_json = os.path.join(gat_output_dir, "training_config.txt"), os.path.join(gat_output_dir, "training_config.json")
    if not os.path.exists(dst_txt) and not os.path.exists(dst_json):
        legacy_txt = os.path.join(os.path.dirname(abs_src), "training_config.txt")
        if os.path.exists(legacy_txt):
            shutil.copy2(legacy_txt, dst_txt)

    info_path = os.path.join(gat_output_dir, "pretrained_checkpoint_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(
            f"""================================================================================
COPIED PRE-TRAINED GAT CHECKPOINT NOTICE
================================================================================
Status:                       COPIED FROM PRE-TRAINED MODEL (Not trained in this run)
Original Pretrained GAT Path: {abs_src}
Original Relative Path:       {to_rel_path(abs_src)}
Cloning Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Note:                         'training_config.json' / 'training_config.txt' in this folder
                              reflect the original GAT pre-training configuration.
================================================================================

[DOWNSTREAM PATIENT RUN CONFIGURATION THAT IMPORTED THIS CHECKPOINT]
{config_text}"""
        )
    logging.info(f"Pre-trained checkpoint notice saved to: {info_path}\n")