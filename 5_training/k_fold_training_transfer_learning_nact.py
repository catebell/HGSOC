import os
import inspect
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from torch_geometric.loader import DataLoader

import config_file_tree
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from PatientBagDataset import PatientBagDataset
from models.GAT import GAT
from models.PatientMIL_regularized import PatientMILModel
from utils_train_eval import pre_train_imputation_encoding
from train_functions import train_epoch_mil, evaluate_mil


# =============================================================================
# CONFIGURAZIONE PERCORSI PDS SORGENTE E OUTPUT NACT
# =============================================================================
PDS_RUN_DIR = os.path.join("trained_models", "18_11_2026_hybrid_MIL_platRes_PDS_fixed")
PDS_GAT_DIR = os.path.join(PDS_RUN_DIR, "GAT")
PDS_MIL_DIR = os.path.join(PDS_RUN_DIR, "MIL")

use_metadata = config_file_tree.USE_SAMPLES_METADATA
use_clinical = config_file_tree.USE_CLINICAL_DATA
sample_target_col = config_file_tree.SAMPLE_TARGET_COL
patient_target_col = config_file_tree.PATIENT_TARGET_COL

PDS_GAT_CKPT = os.path.join(PDS_GAT_DIR, f"best_GAT_{sample_target_col}_{'withMeta' if use_metadata else 'noMeta'}.pt")
PDS_MIL_CKPT = os.path.join(PDS_MIL_DIR, f"best_MIL_{patient_target_col}_{'withClin' if use_clinical else 'noClin'}.pt")
PDS_NODE_SCALER = os.path.join(PDS_GAT_DIR, f"node_scaler_{sample_target_col}.joblib")
PDS_META_PREP = os.path.join(PDS_GAT_DIR, f"metadata_preprocessors_{sample_target_col}.joblib")
PDS_CLIN_PREP = os.path.join(PDS_MIL_DIR, f"clinical_preprocessors_{patient_target_col}.joblib")
PDS_LE_SAMPLE = os.path.join(PDS_GAT_DIR, f"label_encoder_{sample_target_col}.joblib")
PDS_LE_PATIENT = os.path.join(PDS_MIL_DIR, f"label_encoder_patient_{patient_target_col}.joblib")

# File NACT generati da 1_tissues_selection_and_split_separate_nact.py
NACT_CLIN_PATH = "clinical_data_nact.tsv" if os.path.exists("clinical_data_nact.tsv") else os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_nact.tsv")
NACT_ALL_SAMPLES = "samples_nact.csv" if os.path.exists("samples_nact.csv") else os.path.join(config_file_tree.SPLIT_DIR, "samples_nact.csv")
NACT_TV_SAMPLES = "samples_nact_train_val.csv" if os.path.exists("samples_nact_train_val.csv") else os.path.join(config_file_tree.SPLIT_DIR, "samples_nact_train_val.csv")
NACT_TEST_SAMPLES = "samples_nact_test.csv" if os.path.exists("samples_nact_test.csv") else os.path.join(config_file_tree.SPLIT_DIR, "samples_nact_test.csv")

curr_date = datetime.now().strftime("%d_%m_%Y_transfer_PDS_to_NACT")
OUTPUT_DIR = os.path.join("trained_models", curr_date)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "transfer_pds_to_nact.log"), mode="w"),
        logging.StreamHandler()
    ],
    force=True
)


def build_models(num_node_features, num_nodes, num_sample_classes, num_sample_feats,
                 num_patient_classes, num_clin_feats, device):
    """Istanzia GAT e PatientMILModel compatibili con i checkpoint salvati su PDS."""
    gat = GAT(
        num_node_features=num_node_features,
        num_classes=num_sample_classes,
        num_nodes=num_nodes,
        num_sample_features=num_sample_feats if use_metadata else 0,
    ).to(device)

    mil_kwargs = dict(
        sample_encoder=gat,
        num_classes=num_patient_classes,
        num_tissue_classes=num_sample_classes,
        proj_dim=24,
        pooling_hidden_dim=16,
        classifier_hidden_dim=16,
        num_clinical_features=num_clin_feats if use_clinical else 0,
        dropout=0.3,
        subsample_rate=1.0,
    )
    if "num_node_features" in inspect.signature(PatientMILModel.__init__).parameters:
        mil_kwargs["num_node_features"] = num_node_features

    mil = PatientMILModel(**mil_kwargs).to(device)
    return gat, mil


def apply_pds_preprocessors(graphs_list, df_samples, df_clin, le_sample, node_scaler,
                            meta_prep=None, clin_prep=None):
    """Applica esattamente lo scaling e l'encoding fittati su PDS (zero data leakage)."""
    for g in graphs_list:
        g.x = torch.tensor(node_scaler.transform(g.x_raw.numpy()), dtype=torch.float32)

    if use_metadata and meta_prep is not None:
        df_meta_in = df_samples.copy()
        df_meta_in[sample_target_col] = le_sample.transform(df_meta_in[sample_target_col])
        df_meta_proc = pre_train_imputation_encoding(
            df=df_meta_in, task="classification", target_col=sample_target_col, is_train=False,
            encoders=meta_prep["encoders"], scalers=meta_prep["scalers"],
            imputers=meta_prep["imputers"], categories=meta_prep["categories"]
        ).set_index("sample_id")
        sample_cols = [c for c in df_meta_proc.columns if c not in {sample_target_col, "patient_id"}]
        update_graphs_metadata(graphs_list, df_meta_proc, sample_cols)

    patient_clinical_map = None
    num_clin_feats = 0
    if use_clinical and clin_prep is not None:
        df_clin_proc = pre_train_imputation_encoding(
            df=df_clin.copy(), task="classification", target_col=patient_target_col, is_train=False,
            encoders=clin_prep["encoders"], scalers=clin_prep["scalers"],
            imputers=clin_prep["imputers"], categories=clin_prep["categories"]
        ).set_index("patient_id")
        clin_cols = [c for c in df_clin_proc.columns if c not in {patient_target_col, "patient_id"}]
        num_clin_feats = len(clin_cols)
        patient_clinical_map = {
            pid: torch.tensor(df_clin_proc.loc[pid, clin_cols].values.astype(np.float32), dtype=torch.float32)
            for pid in df_clin_proc.index
        }

    return patient_clinical_map, num_clin_feats


@torch.no_grad()
def predict_probs(device, model, loader):
    """Ritorna patient_ids, y_true e probabilità P(y=1) per ogni paziente."""
    model.eval()
    pids, ys, probs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        p1 = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs.extend(p1.tolist())
        ys.extend(batch.patient_y.cpu().numpy().tolist())
        pids.extend([pid[0] if isinstance(pid, (list, tuple)) else pid for pid in batch.patient_id])
    return np.array(pids), np.array(ys), np.array(probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device} | Sorgente PDS: {PDS_RUN_DIR}")

    # 1. Caricamento artefatti PDS
    le_sample = joblib.load(PDS_LE_SAMPLE)
    le_patient = joblib.load(PDS_LE_PATIENT)
    node_scaler = joblib.load(PDS_NODE_SCALER)
    meta_prep = joblib.load(PDS_META_PREP) if (use_metadata and os.path.exists(PDS_META_PREP)) else None
    clin_prep = joblib.load(PDS_CLIN_PREP) if (use_clinical and os.path.exists(PDS_CLIN_PREP)) else None

    # 2. Caricamento dataset NACT
    df_clin_nact = pd.read_csv(NACT_CLIN_PATH, sep="\t").dropna(subset=[patient_target_col]).copy()
    df_clin_nact[patient_target_col] = le_patient.transform(df_clin_nact[patient_target_col])
    patient_target_map = df_clin_nact.set_index("patient_id")[patient_target_col].to_dict()

    df_samples_all = pd.read_csv(NACT_ALL_SAMPLES)
    df_samples_all = df_samples_all[df_samples_all["patient_id"].isin(patient_target_map)].copy()
    df_samples_enc = df_samples_all.copy()
    df_samples_enc[sample_target_col] = le_sample.transform(df_samples_enc[sample_target_col])

    raw_ds = PatientSampleDataset(
        config_file_tree.PROCESSED_GRAPHS_DIR, df_samples_enc,
        task="classification", target_col=sample_target_col
    )
    all_graphs = []
    for i in range(len(raw_ds)):
        g = raw_ds[i]
        g.x_raw = g.x.clone()
        all_graphs.append(g)

    patient_clin_map, num_clin_feats = apply_pds_preprocessors(
        all_graphs, df_samples_all, df_clin_nact, le_sample, node_scaler, meta_prep, clin_prep
    )
    num_sample_feats = all_graphs[0].metadata_x.shape[-1] if (use_metadata and hasattr(all_graphs[0], "metadata_x")) else 0

    gat_base, mil_base = build_models(
        num_node_features=raw_ds.num_features,
        num_nodes=all_graphs[0].x.shape[0],
        num_sample_classes=len(le_sample.classes_),
        num_sample_feats=num_sample_feats,
        num_patient_classes=len(le_patient.classes_),
        num_clin_feats=num_clin_feats,
        device=device
    )
    gat_base.load_state_dict(torch.load(PDS_GAT_CKPT, map_location=device))
    mil_base.load_state_dict(torch.load(PDS_MIL_CKPT, map_location=device))

    # =========================================================================
    # FASE A: VALUTAZIONE ZERO-SHOT SU TUTTA LA COORTE NACT (N=78)
    # =========================================================================
    bag_all_nact = PatientBagDataset(all_graphs, patient_target_map, patient_clinical_map=patient_clin_map)
    loader_all_nact = DataLoader(bag_all_nact, batch_size=8, shuffle=False)

    pids_zs, y_zs, probs_zs = predict_probs(device, mil_base, loader_all_nact)
    preds_zs = (probs_zs >= 0.5).astype(int)

    # Correzione analitica del prior shift (PDS ~23% yes -> NACT ~64% yes)
    pi_pds = 0.2302
    pi_nact = float(y_zs.mean())
    delta_logit = np.log(pi_nact / (1.0 - pi_nact)) - np.log(pi_pds / (1.0 - pi_pds))
    logits_zs = np.log(np.clip(probs_zs, 1e-6, 1 - 1e-6) / (1.0 - np.clip(probs_zs, 1e-6, 1 - 1e-6)))
    probs_zs_cal = 1.0 / (1.0 + np.exp(-(logits_zs + delta_logit)))
    preds_zs_cal = (probs_zs_cal >= 0.5).astype(int)

    logging.info("\n" + "=" * 65)
    logging.info(f"FASE A - ZERO-SHOT PDS -> NACT (N={len(y_zs)} pazienti)")
    logging.info(f"Zero-Shot Raw   | AUC: {roc_auc_score(y_zs, probs_zs):.4f} | "
                 f"BalAcc: {balanced_accuracy_score(y_zs, preds_zs):.4f} | "
                 f"F1: {f1_score(y_zs, preds_zs, average='macro'):.4f} | Acc: {accuracy_score(y_zs, preds_zs):.4f}")
    logging.info(f"Zero-Shot Prior | AUC: {roc_auc_score(y_zs, probs_zs_cal):.4f} | "
                 f"BalAcc: {balanced_accuracy_score(y_zs, preds_zs_cal):.4f} | "
                 f"F1: {f1_score(y_zs, preds_zs_cal, average='macro'):.4f} | Acc: {accuracy_score(y_zs, preds_zs_cal):.4f}")
    logging.info("=" * 65 + "\n")

    # =========================================================================
    # FASE B: FINE-TUNING GRADUALE 5-FOLD SU NACT TRAIN+VAL + HOLD-OUT TEST
    # =========================================================================
    tv_sample_ids = set(pd.read_csv(NACT_TV_SAMPLES)["sample_id"])
    te_sample_ids = set(pd.read_csv(NACT_TEST_SAMPLES)["sample_id"])

    tv_graphs = [g for g in all_graphs if g.sample_id in tv_sample_ids]
    te_graphs = [g for g in all_graphs if g.sample_id in te_sample_ids]

    tv_groups = np.array([g.patient_id for g in tv_graphs])
    tv_y_samples = np.array([patient_target_map[g.patient_id] for g in tv_graphs])

    bag_test = PatientBagDataset(te_graphs, patient_target_map, patient_clinical_map=patient_clin_map)
    loader_test = DataLoader(bag_test, batch_size=8, shuffle=False)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    oof_records = []
    test_probs_folds = []

    warmup_epochs = 8
    total_epochs = 25
    top_gat_layers = ("conv2", "norm2", "readout")

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(tv_graphs, tv_y_samples, groups=tv_groups)):
        logging.info(f"--- FINE-TUNING FOLD {fold + 1}/5 ---")
        _, mil_ft = build_models(
            num_node_features=raw_ds.num_features,
            num_nodes=all_graphs[0].x.shape[0],
            num_sample_classes=len(le_sample.classes_),
            num_sample_feats=num_sample_feats,
            num_patient_classes=len(le_patient.classes_),
            num_clin_feats=num_clin_feats,
            device=device
        )
        mil_ft.load_state_dict(torch.load(PDS_MIL_CKPT, map_location=device))

        # Aggiusta il bias del classificatore finale dal prior PDS al prior NACT
        with torch.no_grad():
            last_linear = mil_ft.classifier[-1]
            if isinstance(last_linear, torch.nn.Linear) and last_linear.bias is not None:
                last_linear.bias[1] += float(delta_logit)

        train_ds = PatientBagDataset([tv_graphs[i] for i in tr_idx], patient_target_map, patient_clinical_map=patient_clin_map)
        val_ds = PatientBagDataset([tv_graphs[i] for i in va_idx], patient_target_map, patient_clinical_map=patient_clin_map)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        tr_targets = torch.tensor([patient_target_map[pid] for pid in train_ds.patient_ids], dtype=torch.long)
        counts = torch.clamp(torch.bincount(tr_targets, minlength=2), min=1)
        weights = (len(tr_targets) / (2.0 * counts.float())).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        mil_ft.freeze_sample_encoder()
        gat_params = set(mil_ft.sample_encoder.parameters())
        head_params = [p for p in mil_ft.parameters() if p not in gat_params]
        top_gat_params = [
            p for lname in top_gat_layers if hasattr(mil_ft.sample_encoder, lname)
            for p in getattr(mil_ft.sample_encoder, lname).parameters()
        ]

        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": 3e-4, "weight_decay": 3e-2},
            {"params": top_gat_params, "lr": 2e-6, "weight_decay": 1e-2},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        best_score = float("inf")
        ema_loss = None
        best_weights = None
        patience_cnt = 0

        for epoch in range(1, total_epochs + 1):
            if epoch == warmup_epochs + 1:
                mil_ft.unfreeze_top_gat_layers(top_gat_layers)
                patience_cnt = 0

            tr_loss, _ = train_epoch_mil(device, mil_ft, train_loader, optimizer, criterion)
            va_loss, va_metrics, _ = evaluate_mil(device, mil_ft, val_loader, criterion)
            scheduler.step()

            ema_loss = va_loss if ema_loss is None else (0.6 * va_loss + 0.4 * ema_loss)
            score = ema_loss - 0.15 * va_metrics["auc"]

            if epoch >= 3 and score < best_score:
                best_score = score
                best_weights = {k: v.cpu().clone() for k, v in mil_ft.state_dict().items()}
                patience_cnt = 0
            elif epoch > warmup_epochs:
                patience_cnt += 1

            if patience_cnt > 8:
                break

        if best_weights is not None:
            mil_ft.load_state_dict(best_weights)

        va_pids, va_y, va_probs = predict_probs(device, mil_ft, val_loader)
        for pid, yt, yp in zip(va_pids, va_y, va_probs):
            oof_records.append({"patient_id": pid, "y_true": yt, "prob": yp})

        te_pids, te_y, te_probs = predict_probs(device, mil_ft, loader_test)
        test_probs_folds.append(te_probs)
        logging.info(f"Fold {fold + 1} | Val AUC: {roc_auc_score(va_y, va_probs):.4f} | Test AUC: {roc_auc_score(te_y, te_probs):.4f}")

    # Riepilogo OOF e Test Ensemble
    df_oof = pd.DataFrame(oof_records).groupby("patient_id", as_index=False).mean()
    oof_preds = (df_oof["prob"] >= 0.5).astype(int)

    ens_test_probs = np.mean(test_probs_folds, axis=0)
    ens_test_preds = (ens_test_probs >= 0.5).astype(int)

    logging.info("\n" + "=" * 65)
    logging.info("FASE B - RISULTATI FINE-TUNING PDS -> NACT")
    logging.info(f"OOF Train+Val (N={len(df_oof)}) | AUC: {roc_auc_score(df_oof['y_true'], df_oof['prob']):.4f} | "
                 f"BalAcc: {balanced_accuracy_score(df_oof['y_true'], oof_preds):.4f} | "
                 f"F1: {f1_score(df_oof['y_true'], oof_preds, average='macro'):.4f}")
    logging.info(f"Hold-out Test (N={len(te_y)})   | AUC: {roc_auc_score(te_y, ens_test_probs):.4f} | "
                 f"BalAcc: {balanced_accuracy_score(te_y, ens_test_preds):.4f} | "
                 f"F1: {f1_score(te_y, ens_test_preds, average='macro'):.4f}")
    logging.info("=" * 65)

    df_oof.to_csv(os.path.join(OUTPUT_DIR, "nact_transfer_oof_predictions.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()