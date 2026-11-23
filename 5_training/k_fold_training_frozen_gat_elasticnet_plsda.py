import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from train_functions import (
    SelectiveNodeScaler,
    aggregate_patient_bags,
    compute_gat_gene_attribution_exact_bag,
    compute_plsda_vip_scores,
    deconfound_tissue_embeddings,
    extract_frozen_gat_sample_embeddings,
    load_gene_node_mapping,
    show_fold_results,
)
from utils_train_eval import (
    handle_pretrained_gat_checkpoint,
    pre_train_imputation_encoding,
    print_details,
    save_training_config_report,
)

# =============================================================================
# 1. EXPERIMENT CONFIGURATION & DIRECTORY SETUP
# =============================================================================
PATIENT_MODEL_SUBDIR = "LinearMIL"
USE_TISSUE_DECONFOUNDING = True  # Toggle tissue-centroid centering + L2 normalization
POOLING_MODE = "softmax"
POOLING_TAU = 0.08

use_metadata = config_file_tree.USE_SAMPLES_METADATA
use_clinical = config_file_tree.USE_CLINICAL_DATA
task = config_file_tree.TASK
sample_target_col = config_file_tree.SAMPLE_TARGET_COL
patient_target_col = config_file_tree.PATIENT_TARGET_COL

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
path_to_clinical = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")
path_train_val_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_train_val.csv")
path_test_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")
path_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR

gat_pretrained_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
pretrained_encoder_path = os.path.join(
    gat_pretrained_dir,
    f"best_GAT_{sample_target_col}_{'withMeta' if use_metadata else 'noMeta'}.pt",
)

deconf_tag = "tissueDeconf" if USE_TISSUE_DECONFOUNDING else "noTissueDeconf"
clin_tag = "withClin" if use_clinical else "noClin"
run_name = datetime.now().strftime(f"%d_%m_%Y_LinearMIL_{patient_target_col}_{deconf_tag}_{clin_tag}_PDS")
output_base_dir = os.path.join("trained_models", run_name)
os.makedirs(output_base_dir, exist_ok=True)

# Save configuration directly inside LinearMIL/ (never creating an unused MIL/ folder)
_, config_text = save_training_config_report(output_base_dir, patient_subdir=PATIENT_MODEL_SUBDIR)

gat_output_dir = os.path.join(output_base_dir, "GAT")
handle_pretrained_gat_checkpoint(gat_pretrained_dir, gat_output_dir, config_text)

linear_mil_dir = os.path.join(output_base_dir, PATIENT_MODEL_SUBDIR)
emb_output_dir = os.path.join(linear_mil_dir, "patient_embeddings")
fold_models_dir = os.path.join(linear_mil_dir, "fold_models")
os.makedirs(emb_output_dir, exist_ok=True)
os.makedirs(fold_models_dir, exist_ok=True)

active_feature_names = [config_file_tree.NODE_FEATURE_NAMES[i] for i in config_file_tree.NODE_FEATURE_INDEX]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(output_base_dir, "last_execution.log"), mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(
    f"Device: {device} | Deconfounding: {USE_TISSUE_DECONFOUNDING} | "
    f"USE_CLINICAL_DATA: {use_clinical} | USE_SAMPLES_METADATA: {use_metadata}\n"
)

df_gene_mapping = load_gene_node_mapping()

# =============================================================================
# 2. DATA LOADING & TARGET ENCODING
# =============================================================================
df_metadata = pd.read_csv(path_to_metadata, sep="\t")
train_val_ids = set(pd.read_csv(path_train_val_csv)["sample_id"])
test_ids = set(pd.read_csv(path_test_csv)["sample_id"])

df_metadata_tv = df_metadata[df_metadata["sample_id"].isin(train_val_ids)].dropna(subset=[sample_target_col]).copy()
df_metadata_test = df_metadata[df_metadata["sample_id"].isin(test_ids)].dropna(subset=[sample_target_col]).copy()

cols_to_keep = ["patient_id", patient_target_col]
if use_clinical:
    cols_to_keep += [c for c in config_file_tree.CLINICAL_FEATURES if c not in cols_to_keep]
df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
df_clinical["patient_id"] = df_clinical["patient_id"].astype(str).str.strip()

sample_le_path = os.path.join(gat_output_dir, f"label_encoder_{sample_target_col}.joblib")
if os.path.exists(sample_le_path):
    label_encoder_sample = joblib.load(sample_le_path)
else:
    label_encoder_sample = LabelEncoder().fit(df_metadata[sample_target_col].dropna())
    joblib.dump(label_encoder_sample, sample_le_path)

df_metadata_tv[sample_target_col] = label_encoder_sample.transform(df_metadata_tv[sample_target_col])
df_metadata_test[sample_target_col] = label_encoder_sample.transform(df_metadata_test[sample_target_col])
num_classes_samples = len(label_encoder_sample.classes_)

df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).drop_duplicates(subset=["patient_id"]).copy()
label_encoder_patient = LabelEncoder()
df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])
joblib.dump(label_encoder_patient, os.path.join(linear_mil_dir, f"label_encoder_patient_{patient_target_col}.joblib"))
logging.info(f"Patient classes ({patient_target_col}): {label_encoder_patient.classes_}\n")

patient_target_map = df_clinical_clean.set_index("patient_id")[patient_target_col].to_dict()

raw_tv_ds = PatientSampleDataset(path_graphs_dir, df_metadata_tv, task=task, target_col=sample_target_col)
raw_test_ds = PatientSampleDataset(path_graphs_dir, df_metadata_test, task=task, target_col=sample_target_col)

train_val_dataset = [raw_tv_ds[i] for i in tqdm(range(len(raw_tv_ds)), desc="Loading Train/Val")]
test_dataset = [raw_test_ds[i] for i in tqdm(range(len(raw_test_ds)), desc="Loading Test")]
print_details(raw_tv_ds)

num_genes_features = raw_tv_ds.num_features
num_nodes_per_graph = raw_tv_ds[0].x.shape[0]
for g in train_val_dataset + test_dataset:
    g.x_raw = g.x.clone()

# Optional clinical annotations for post-hoc patient embedding inspection
df_clinical_annot = None
full_clin_path = os.path.join(config_file_tree.PREPROCESSED_DATA_DIR, "preprocessed_clinical_data.tsv")
if os.path.exists(full_clin_path):
    df_clin_raw = pd.read_csv(full_clin_path, sep="\t")
    if "patient_id" in df_clin_raw.columns:
        df_clin_raw["patient_id"] = df_clin_raw["patient_id"].astype(str).str.strip()
        wanted_cols = ["patient_id", "Treatment_strategy", "PFI_KaplanM_allHGSC", "Residual_tumor", "is_hrd"]
        df_clinical_annot = df_clin_raw[[c for c in wanted_cols if c in df_clin_raw.columns]].drop_duplicates("patient_id")


def _binary_metrics(y_true, y_pred, y_prob, **extra):
    """Compute standard binary classification metrics for a fold."""
    return {
        **extra,
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


# =============================================================================
# 3. STRATIFIED GROUP 5-FOLD CROSS-VALIDATION
# =============================================================================
groups = df_metadata_tv["patient_id"].values
composite_y = [
    f"{row[sample_target_col]}_{patient_target_map.get(str(row['patient_id']).strip(), 'unfiltered')}"
    for _, row in df_metadata_tv.iterrows()
]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)

enet_test_results, plsda_test_results, oof_records = [], [], []
enet_full_coef_folds, plsda_full_vip_folds = [], []
test_probs_enet_folds, test_probs_pls_folds = [], []
all_graphs_for_attr, global_max_sample_map = [], {}
gat_model_ref = None

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, composite_y, groups=groups)):
    fold_num = fold + 1
    logging.info(f"================ FOLD {fold_num} ================\n")

    # 3.1 Fit node feature scaler strictly on training split to prevent leakage
    train_nodes_np = np.vstack([train_val_dataset[i].x_raw.numpy() for i in train_idx])
    node_scaler = SelectiveNodeScaler(active_feature_names=active_feature_names).fit(train_nodes_np)

    for idx_tv in np.concatenate([train_idx, val_idx]):
        train_val_dataset[idx_tv].x = torch.tensor(
            node_scaler.transform(train_val_dataset[idx_tv].x_raw.numpy()), dtype=torch.float32
        )
    for g in test_dataset:
        g.x = torch.tensor(node_scaler.transform(g.x_raw.numpy()), dtype=torch.float32)

    # 3.2 Optional sample-level metadata preprocessing
    num_sample_features = 0
    if use_metadata:
        df_tr_p, enc, sca, imp, cat = pre_train_imputation_encoding(
            df_metadata_tv.iloc[train_idx].copy(), task=task, target_col=sample_target_col, is_train=True
        )
        df_va_p = pre_train_imputation_encoding(
            df_metadata_tv.iloc[val_idx].copy(), task=task, target_col=sample_target_col,
            is_train=False, encoders=enc, scalers=sca, imputers=imp, categories=cat
        )
        df_te_p = pre_train_imputation_encoding(
            df_metadata_test.copy(), task=task, target_col=sample_target_col,
            is_train=False, encoders=enc, scalers=sca, imputers=imp, categories=cat
        )
        df_tv_p = pd.concat([df_tr_p, df_va_p], axis=0).set_index("sample_id")
        df_te_p = df_te_p.set_index("sample_id")
        sample_feat_cols = [c for c in df_tv_p.columns if c not in {sample_target_col, "patient_id"}]
        num_sample_features = len(sample_feat_cols)

        update_graphs_metadata(train_val_dataset, df_tv_p, sample_feat_cols)
        update_graphs_metadata(test_dataset, df_te_p, sample_feat_cols)

    # 3.3 Extract frozen GAT sample embeddings, optionally de-confound by tissue, and aggregate patient bags
    model_gat = GAT(
        num_node_features=num_genes_features,
        num_classes=num_classes_samples,
        num_nodes=num_nodes_per_graph,
        num_sample_features=num_sample_features,
    ).to(device)
    model_gat.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
    model_gat.eval()
    if gat_model_ref is None:
        gat_model_ref = model_gat

    df_samp_train = extract_frozen_gat_sample_embeddings(
        device, model_gat, DataLoader(Subset(train_val_dataset, train_idx), batch_size=16, shuffle=False)
    )
    df_samp_val = extract_frozen_gat_sample_embeddings(
        device, model_gat, DataLoader(Subset(train_val_dataset, val_idx), batch_size=16, shuffle=False)
    )
    df_samp_test = extract_frozen_gat_sample_embeddings(
        device, model_gat, DataLoader(test_dataset, batch_size=16, shuffle=False)
    )

    gat_emb_cols = [c for c in df_samp_train.columns if c.startswith("gat_emb_")]
    if USE_TISSUE_DECONFOUNDING:
        df_samp_tr_in, df_samp_va_in, df_samp_te_in = deconfound_tissue_embeddings(
            df_samp_train, df_samp_val, df_samp_test, emb_cols=gat_emb_cols, tissue_col="tissue_label", scale_by_tissue=False
        )
    else:
        df_samp_tr_in, df_samp_va_in, df_samp_te_in = df_samp_train, df_samp_val, df_samp_test

    df_pat_train, _ = aggregate_patient_bags(
        df_samp_tr_in, gat_emb_cols, patient_target_map, label_encoder_sample, pooling_mode=POOLING_MODE, tau=POOLING_TAU
    )
    df_pat_val, val_pool_map = aggregate_patient_bags(
        df_samp_va_in, gat_emb_cols, patient_target_map, label_encoder_sample, pooling_mode=POOLING_MODE, tau=POOLING_TAU
    )
    df_pat_test, test_pool_map = aggregate_patient_bags(
        df_samp_te_in, gat_emb_cols, patient_target_map, label_encoder_sample, pooling_mode=POOLING_MODE, tau=POOLING_TAU
    )

    global_max_sample_map.update(val_pool_map)
    if fold == 0:
        global_max_sample_map.update(test_pool_map)
        all_graphs_for_attr.extend(test_dataset)
    all_graphs_for_attr.extend([train_val_dataset[i] for i in val_idx])

    genomic_feat_cols = [c for c in df_pat_train.columns if c.startswith(("mean_emb_", "max_emb_"))]
    active_model_cols = list(genomic_feat_cols)
    clinical_feat_cols, clin_prep_bundle = [], None

    # 3.4 Fold-specific clinical feature preprocessing (when enabled)
    if use_clinical:
        df_clin_tr_p, p_enc, p_sca, p_imp, p_cat = pre_train_imputation_encoding(
            df_clinical_clean[df_clinical_clean["patient_id"].isin(set(df_pat_train["patient_id"]))].copy(),
            task="classification", target_col=patient_target_col, is_train=True,
        )
        df_clin_va_p = pre_train_imputation_encoding(
            df_clinical_clean[df_clinical_clean["patient_id"].isin(set(df_pat_val["patient_id"]))].copy(),
            task="classification", target_col=patient_target_col, is_train=False,
            encoders=p_enc, scalers=p_sca, imputers=p_imp, categories=p_cat,
        )
        df_clin_te_p = pre_train_imputation_encoding(
            df_clinical_clean[df_clinical_clean["patient_id"].isin(set(df_pat_test["patient_id"]))].copy(),
            task="classification", target_col=patient_target_col, is_train=False,
            encoders=p_enc, scalers=p_sca, imputers=p_imp, categories=p_cat,
        )

        cand_cols = [c for c in df_clin_tr_p.columns if c not in {patient_target_col, "patient_id"}]
        clinical_feat_cols = [c for c in cand_cols if df_clin_tr_p[c].nunique(dropna=True) > 1]
        logging.info(f"[Fold {fold_num}] Active clinical features ({len(clinical_feat_cols)}): {clinical_feat_cols}")

        if clinical_feat_cols:
            df_pat_train = df_pat_train.merge(df_clin_tr_p[["patient_id"] + clinical_feat_cols], on="patient_id", how="left")
            df_pat_val = df_pat_val.merge(df_clin_va_p[["patient_id"] + clinical_feat_cols], on="patient_id", how="left")
            df_pat_test = df_pat_test.merge(df_clin_te_p[["patient_id"] + clinical_feat_cols], on="patient_id", how="left")
            active_model_cols = genomic_feat_cols + clinical_feat_cols
            clin_prep_bundle = {"encoders": p_enc, "scalers": p_sca, "imputers": p_imp, "categories": p_cat}

    X_train, y_train = df_pat_train[active_model_cols].values, df_pat_train["target_label"].values
    X_val, y_val = df_pat_val[active_model_cols].values, df_pat_val["target_label"].values
    X_test, y_test = df_pat_test[active_model_cols].values, df_pat_test["target_label"].values

    # 3.5 Model A: ElasticNet Logistic Regression
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", class_weight="balanced", max_iter=5000, random_state=42)),
    ])
    grid_enet = GridSearchCV(
        enet_pipe,
        param_grid={"clf__C": [0.15, 0.25, 0.5, 1.0], "clf__l1_ratio": [0.2, 0.4]},
        scoring="roc_auc",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42),
        n_jobs=-1,
    ).fit(X_train, y_train)
    best_enet = grid_enet.best_estimator_

    full_coef = best_enet.named_steps["clf"].coef_.ravel()
    enet_full_coef_folds.append(full_coef)

    val_probs_enet = best_enet.predict_proba(X_val)[:, 1]
    val_preds_enet = best_enet.predict(X_val)
    test_probs_enet = best_enet.predict_proba(X_test)[:, 1]
    test_preds_enet = best_enet.predict(X_test)
    test_probs_enet_folds.append(test_probs_enet)

    enet_metrics = _binary_metrics(
        y_test, test_preds_enet, test_probs_enet,
        fold=fold_num,
        best_C=grid_enet.best_params_["clf__C"],
        best_l1_ratio=grid_enet.best_params_["clf__l1_ratio"],
        n_nonzero_feats=int(np.sum(np.abs(full_coef) > 1e-5)),
    )
    enet_test_results.append(enet_metrics)
    logging.info(
        f"[ElasticNet] Fold {fold_num} (C={enet_metrics['best_C']}, l1={enet_metrics['best_l1_ratio']}, "
        f"active={enet_metrics['n_nonzero_feats']}/{len(active_model_cols)}) | "
        f"Test Acc: {enet_metrics['acc']:.4f} | F1: {enet_metrics['f1']:.4f} | AUC: {enet_metrics['auc']:.4f}"
    )

    # 3.6 Model B: 2-Component PLS-DA + Balanced Logistic Head
    scaler_pls = StandardScaler()
    X_train_std = scaler_pls.fit_transform(X_train)
    X_val_std = scaler_pls.transform(X_val)
    X_test_std = scaler_pls.transform(X_test)

    pls = PLSRegression(n_components=2, scale=False).fit(X_train_std, y_train)
    plsda_full_vip_folds.append(compute_plsda_vip_scores(pls))

    T_train, T_val, T_test = pls.transform(X_train_std), pls.transform(X_val_std), pls.transform(X_test_std)
    pls_clf = LogisticRegression(class_weight="balanced", C=1.0, random_state=42).fit(T_train, y_train)

    val_probs_pls = pls_clf.predict_proba(T_val)[:, 1]
    val_preds_pls = pls_clf.predict(T_val)
    test_probs_pls = pls_clf.predict_proba(T_test)[:, 1]
    test_preds_pls = pls_clf.predict(T_test)
    test_probs_pls_folds.append(test_probs_pls)

    pls_metrics = _binary_metrics(y_test, test_preds_pls, test_probs_pls, fold=fold_num)
    plsda_test_results.append(pls_metrics)
    logging.info(
        f"[PLS-DA 2D]  Fold {fold_num} | "
        f"Test Acc: {pls_metrics['acc']:.4f} | F1: {pls_metrics['f1']:.4f} | AUC: {pls_metrics['auc']:.4f}\n"
    )

    for idx_v, row_v in df_pat_val.iterrows():
        oof_records.append({
            "fold": fold_num,
            "patient_id": row_v["patient_id"],
            "target_label": int(row_v["target_label"]),
            "target_label_name": label_encoder_patient.classes_[int(row_v["target_label"])],
            "n_samples": row_v["n_samples"],
            "tissue_types": row_v["tissue_types"],
            "enet_prob_resistant": float(val_probs_enet[idx_v]),
            "enet_pred": int(val_preds_enet[idx_v]),
            "pls_prob_resistant": float(val_probs_pls[idx_v]),
            "pls_pred": int(val_preds_pls[idx_v]),
            "PLS_1": float(T_val[idx_v, 0]),
            "PLS_2": float(T_val[idx_v, 1]),
        })

    # 3.7 Save patient embeddings, 2D PCA/PLS-DA projection plot, and fold artifacts
    for df_sp, sp_name, pls_xy, e_probs in [
        (df_pat_train, "train", T_train, best_enet.predict_proba(X_train)[:, 1]),
        (df_pat_val, "val", T_val, val_probs_enet),
        (df_pat_test, "test", T_test, test_probs_enet),
    ]:
        df_sp["split"] = sp_name
        df_sp[["PLS_1", "PLS_2"]] = pls_xy
        df_sp["enet_prob_resistant"] = e_probs

    df_all_fold = pd.concat([df_pat_train, df_pat_val, df_pat_test], axis=0, ignore_index=True)
    df_all_fold["target_label_name"] = [label_encoder_patient.classes_[y] for y in df_all_fold["target_label"]]

    pca = PCA(n_components=2, random_state=42).fit(X_train_std)
    pca_coords = pca.transform(scaler_pls.transform(df_all_fold[active_model_cols].values))
    df_all_fold["PCA_1"], df_all_fold["PCA_2"] = pca_coords[:, 0], pca_coords[:, 1]

    if df_clinical_annot is not None:
        df_all_fold = df_all_fold.merge(df_clinical_annot, on="patient_id", how="left")
    df_all_fold.to_csv(os.path.join(emb_output_dir, f"patient_embeddings_fold_{fold_num}.tsv"), sep="\t", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors, markers = {"no": "#1f77b4", "yes": "#d62728"}, {"train": "o", "val": "s", "test": "^"}
    for ax, x_col, y_col, title in [
        (ax1, "PCA_1", "PCA_2", f"Fold {fold_num} - Bag PCA ({deconf_tag}, {clin_tag})"),
        (ax2, "PLS_1", "PLS_2", f"Fold {fold_num} - Bag PLS-DA ({deconf_tag}, {clin_tag})"),
    ]:
        for sp in ["train", "val", "test"]:
            for cls_name in ["no", "yes"]:
                sub = df_all_fold[(df_all_fold["split"] == sp) & (df_all_fold["target_label_name"] == cls_name)]
                if not sub.empty:
                    ax.scatter(
                        sub[x_col], sub[y_col],
                        c=colors[cls_name], marker=markers[sp],
                        s=85 if sp == "test" else 60, alpha=0.85, edgecolors="k",
                        label=f"{cls_name} ({sp}, n={len(sub)})",
                    )
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(emb_output_dir, f"embeddings_2d_fold_{fold_num}.png"), dpi=300)
    plt.close()

    joblib.dump(
        {
            "fold": fold_num,
            "val_patient_ids": sorted(df_pat_val["patient_id"].astype(str).str.strip().tolist()),
            "use_tissue_deconfounding": USE_TISSUE_DECONFOUNDING,
            "pooling_mode": POOLING_MODE,
            "tau": POOLING_TAU,
            "node_scaler": node_scaler,
            "df_samp_train_raw": df_samp_train[["tissue_label"] + gat_emb_cols],
            "genomic_feat_cols": genomic_feat_cols,
            "clinical_feat_cols": clinical_feat_cols,
            "active_model_cols": active_model_cols,
            "X_train_mean": X_train.mean(axis=0),
            "best_enet": best_enet,
            "scaler_pls": scaler_pls,
            "pls": pls,
            "pls_clf": pls_clf,
            "clin_prep": clin_prep_bundle,
        },
        os.path.join(fold_models_dir, f"fold_{fold_num}_artifacts.joblib"),
    )

# =============================================================================
# 4. FINAL REPORTS: TEST SET, 5-FOLD ENSEMBLE, OOF & GENE ATTRIBUTION
# =============================================================================
logging.info("=== TEST SET RESULTS (PER FOLD) - ELASTICNET ===")
show_fold_results(pd.DataFrame(enet_test_results), "ElasticNet", save_to_csv=True, dir_to_save=linear_mil_dir, filename="report_kfold_test_ElasticNet.csv")

logging.info("=== TEST SET RESULTS (PER FOLD) - PLS-DA ===")
show_fold_results(pd.DataFrame(plsda_test_results), "PLSDA", save_to_csv=True, dir_to_save=linear_mil_dir, filename="report_kfold_test_PLSDA.csv")

y_test_ref = df_pat_test["target_label"].values
ens_enet_prob = np.mean(test_probs_enet_folds, axis=0)
ens_pls_prob = np.mean(test_probs_pls_folds, axis=0)
logging.info(
    f"=== 5-FOLD ENSEMBLE ON TEST SET (N={len(y_test_ref)}) ===\n"
    f"=== 5-FOLD ENSEMBLE ON TEST SET (N={len(y_test_ref)}) ===\n"
    f"  ElasticNet Ensemble | AUC: {roc_auc_score(y_test_ref, ens_enet_prob):.4f} | PR-AUC: {average_precision_score(y_test_ref, ens_enet_prob):.4f}\n"
    f"  PLS-DA Ensemble     | AUC: {roc_auc_score(y_test_ref, ens_pls_prob):.4f} | PR-AUC: {average_precision_score(y_test_ref, ens_pls_prob):.4f}"
)

df_oof = pd.DataFrame(oof_records)
if df_clinical_annot is not None:
    df_oof = df_oof.merge(df_clinical_annot, on="patient_id", how="left")
df_oof.to_csv(os.path.join(linear_mil_dir, "oof_patient_predictions.tsv"), sep="\t", index=False)

y_oof = df_oof["target_label"].values
logging.info(f"\n=== OUT-OF-FOLD (OOF) GLOBAL METRICS (N={len(df_oof)} PATIENTS) ===")
for label, pred_col, prob_col in [
    ("ElasticNet OOF", "enet_pred", "enet_prob_resistant"),
    ("PLS-DA OOF    ", "pls_pred", "pls_prob_resistant"),
]:
    logging.info(
        f"{label} | Acc: {accuracy_score(y_oof, df_oof[pred_col]):.4f} | "
        f"F1: {f1_score(y_oof, df_oof[pred_col], average='macro'):.4f} | "
        f"AUC: {roc_auc_score(y_oof, df_oof[prob_col]):.4f} | "
        f"PR-AUC: {average_precision_score(y_oof, df_oof[prob_col]):.4f}"
    )

mean_full_coef = np.mean(enet_full_coef_folds, axis=0)
mean_full_vip = np.mean(plsda_full_vip_folds, axis=0)
pd.DataFrame({
    "feature": active_model_cols,
    "enet_mean_coef": mean_full_coef,
    "enet_abs_coef": np.abs(mean_full_coef),
    "plsda_mean_vip": mean_full_vip,
}).sort_values("enet_abs_coef", ascending=False).to_csv(
    os.path.join(linear_mil_dir, "latent_dimensions_importance.tsv"), sep="\t", index=False
)

if gat_model_ref is not None and all_graphs_for_attr:
    logging.info("Computing exact bag-level gene attributions across 5 folds...")
    df_gene_attr = compute_gat_gene_attribution_exact_bag(
        device=device,
        model_gat=gat_model_ref,
        graphs_list=all_graphs_for_attr,
        max_sample_map=global_max_sample_map,
        latent_weights_128=mean_full_coef[:len(genomic_feat_cols)],
        df_gene_mapping=df_gene_mapping,
        feature_names=active_feature_names,
    )
    df_gene_attr.to_csv(os.path.join(linear_mil_dir, "frozen_gat_gene_attribution_mapped.tsv"), sep="\t", index=False)
    logging.info(f"Top 15 genes (HGNC) associated with target outcome:\n{df_gene_attr.head(15).to_string(index=False)}\n")