import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

import config_file_tree


def _unwrap_id(val):
    """Unwrap nested lists, tuples, or scalar tensors into a clean stripped string."""
    while isinstance(val, (list, tuple, np.ndarray)):
        val = val[0] if len(val) > 0 else ""
    if hasattr(val, "item"):
        val = val.item()
    return str(val).strip()


def _unpack_mil_batch(item, device):
    """Unpack a MIL DataLoader item (either PatientBatch or tuple) onto the target device."""
    if isinstance(item, (list, tuple)):
        patient_batch = item[0].to(device)
        target_y = item[1].to(device)
        raw_p_ids = item[2] if len(item) > 2 else getattr(patient_batch, "patient_id", "unknown")
    else:
        patient_batch = item.to(device)
        target_y = patient_batch.patient_y
        raw_p_ids = getattr(patient_batch, "patient_id", "unknown")
    return patient_batch, target_y, raw_p_ids


# =============================================================================
# 1. FEATURE SCALING
# =============================================================================
class SelectiveNodeScaler(BaseEstimator, TransformerMixin):
    """Apply modality-specific scaling to each graph node feature."""

    def __init__(self, active_feature_names):
        self.active_feature_names = active_feature_names
        self.scalers = {}

    @staticmethod
    def _build_scaler_for_feature(feature_name):
        """Select the appropriate transformation for each omic feature type."""
        if feature_name in ("gene_expression", "log2CopyNumber"):
            return StandardScaler()
        if feature_name in ("nMajor", "nMinor"):
            return MinMaxScaler()
        if feature_name == "breaksInGene":
            return make_pipeline(FunctionTransformer(np.log1p, validate=False), MinMaxScaler())
        if feature_name in ("Loh", "mutation_anyP"):
            return "passthrough"
        return StandardScaler()

    def fit(self, X, y=None):
        """Fit per-column scalers on node matrix X [N_nodes, N_active_features]."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        self.scalers = {}
        for idx, name in enumerate(self.active_feature_names):
            scaler = self._build_scaler_for_feature(name)
            if scaler != "passthrough":
                scaler.fit(X_np[:, idx : idx + 1])
            self.scalers[idx] = scaler
        return self

    def transform(self, X):
        """Transform node matrix X using fitted per-column scalers."""
        is_tensor = isinstance(X, torch.Tensor)
        X_np = X.detach().cpu().numpy() if is_tensor else np.copy(X)
        X_out = np.zeros_like(X_np, dtype=np.float32)

        for idx, scaler in self.scalers.items():
            col = X_np[:, idx : idx + 1]
            X_out[:, idx : idx + 1] = col if scaler == "passthrough" else scaler.transform(col)

        return torch.tensor(X_out, dtype=X.dtype, device=X.device) if is_tensor else X_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# =============================================================================
# 2. SAMPLE ENCODER (GAT) TRAINING & EVALUATION
# =============================================================================
def train_epoch(device, model, loader, optimizer, criterion):
    """Train the sample-level GAT encoder for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch, metadata_x=getattr(data, "metadata_x", None))
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
        total += data.num_graphs

    return total_loss / total, correct / total


def evaluate(device, model, loader, criterion):
    """Evaluate the sample-level GAT encoder and return loss + classification metrics."""
    model.eval()
    total_loss, total = 0.0, 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, metadata_x=getattr(data, "metadata_x", None))
            total_loss += criterion(out, data.y).item() * data.num_graphs
            total += data.num_graphs

            all_targets.extend(data.y.cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(out, dim=1).cpu().numpy())

    y_true, y_pred, y_prob = np.array(all_targets), np.array(all_preds), np.array(all_probs)
    n_classes = len(np.unique(y_true))
    auc_val = (
        roc_auc_score(y_true, y_prob[:, 1]) if n_classes == 2
        else (roc_auc_score(y_true, y_prob, multi_class="ovr") if n_classes > 2 else 0.0)
    )

    metrics = {
        "acc": float(np.mean(y_pred == y_true)),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": auc_val,
    }
    return total_loss / total, metrics


# =============================================================================
# 3. NEURAL MULTIPLE INSTANCE LEARNING (MIL) TRAINING & EVALUATION
# =============================================================================
def train_epoch_mil(device, model, loader, optimizer, criterion):
    """Train the end-to-end patient MIL model for one epoch."""
    model.train()
    total_loss, correct, total_patients = 0.0, 0, 0

    for patient_batch in loader:
        patient_batch = patient_batch.to(device)
        target_y = patient_batch.patient_y
        optimizer.zero_grad()

        logits, _ = model(patient_batch)
        loss = criterion(logits, target_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = target_y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == target_y).sum().item()
        total_patients += bs

    return total_loss / total_patients, correct / total_patients


def evaluate_mil(device, model, loader, criterion, return_predictions=False):
    """
    Evaluate the neural MIL model with global and per-class metrics (Precision, Recall,
    F1, ROC-AUC, PR-AUC) and extract per-sample attention weights.
    """
    model.eval()
    total_loss, total_patients = 0.0, 0
    all_targets, all_preds, all_probs, attention_records = [], [], [], []

    with torch.no_grad():
        for item in loader:
            patient_batch, target_y, raw_p_ids = _unpack_mil_batch(item, device)
            logits, attn_weights = model(patient_batch)
            bs = target_y.size(0)

            total_loss += criterion(logits, target_y).item() * bs
            total_patients += bs

            probs = torch.softmax(logits, dim=1)
            all_targets.extend(target_y.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            weights = np.atleast_1d(attn_weights.squeeze().cpu().numpy())
            raw_sids = getattr(patient_batch, "sample_id", None)
            if raw_sids is not None:
                if isinstance(raw_sids, (list, tuple)) and raw_sids and isinstance(raw_sids[0], (list, tuple)):
                    sample_ids = [s for sub in raw_sids for s in sub]
                elif isinstance(raw_sids, (list, tuple)):
                    sample_ids = list(raw_sids)
                elif isinstance(raw_sids, np.ndarray):
                    sample_ids = raw_sids.flatten().tolist()
                else:
                    sample_ids = [raw_sids]
            else:
                sample_ids = [f"sample_{j}" for j in range(len(weights))]

            tissue_labels = (
                patient_batch.y.cpu().numpy().tolist()
                if hasattr(patient_batch, "y") and patient_batch.y is not None
                else [None] * len(weights)
            )

            p_id_list = [_unwrap_id(p) for p in raw_p_ids] if isinstance(raw_p_ids, (list, tuple)) else [_unwrap_id(raw_p_ids)]
            sample_patient_ids = []
            if hasattr(patient_batch, "num_samples") and patient_batch.num_samples is not None:
                ns_list = patient_batch.num_samples.tolist() if hasattr(patient_batch.num_samples, "tolist") else patient_batch.num_samples
                for p_idx, n_s in enumerate(ns_list):
                    curr_pid = p_id_list[p_idx] if p_idx < len(p_id_list) else p_id_list[0]
                    sample_patient_ids.extend([curr_pid] * n_s)

            if len(sample_patient_ids) != len(sample_ids):
                sample_patient_ids = [str(s_id).split("_")[0] for s_id in sample_ids]

            for s_id, p_id_curr, w, t_lab in zip(sample_ids, sample_patient_ids, weights, tissue_labels):
                attention_records.append({
                    "patient_id": p_id_curr,
                    "sample_id": s_id,
                    "tissue_label": t_lab,
                    "attention_weight": float(w),
                })

    y_true, y_pred, y_prob = np.array(all_targets), np.array(all_preds), np.array(all_probs)
    avg_loss = total_loss / total_patients if total_patients > 0 else 0.0

    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        auc_val = roc_auc_score(y_true, y_prob[:, 1])
        pr_auc_val = average_precision_score(y_true, y_prob[:, 1])
    elif n_classes > 2:
        auc_val, pr_auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr"), 0.0
    else:
        auc_val, pr_auc_val = 0.0, 0.0

    prec_cls = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    rec_cls = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1_cls = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": auc_val,
        "pr_auc": pr_auc_val,
        "prec_class_0": float(prec_cls[0]),
        "rec_class_0": float(rec_cls[0]),
        "f1_class_0": float(f1_cls[0]),
        "prec_class_1": float(prec_cls[1]),
        "rec_class_1": float(rec_cls[1]),
        "f1_class_1": float(f1_cls[1]),
    }

    df_attention = pd.DataFrame(attention_records)
    if return_predictions:
        return avg_loss, metrics, df_attention, {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
    return avg_loss, metrics, df_attention


def train_epoch_mil_latent_smote(
    device, model, loader, optimizer, criterion_unweighted,
    k_neighbors=3, random_state=42, epoch=1, warmup_epochs=8, smote_batch_size=16
):
    """
    Two-stage MIL training with latent-space SMOTE oversampling:
      - Epochs 1..warmup_epochs: standard mini-batch training on real patient bags.
      - Epochs > warmup_epochs: (2A) update backbone + attention pooling on real bags;
        (2B) extract clean patient embeddings, oversample minority class via SMOTE,
        and train the classification head on balanced mini-batches.
    """
    from imblearn.over_sampling import SMOTE

    if epoch <= warmup_epochs:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for patient_batch in loader:
            patient_batch = patient_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(patient_batch)
            loss = criterion_unweighted(logits, patient_batch.patient_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = patient_batch.patient_y.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == patient_batch.patient_y).sum().item()
            total += bs
        return total_loss / max(1, total), correct / max(1, total)

    # Stage 2A: Update backbone & attention pooling while keeping classifier fixed
    model.train()
    for p in model.classifier.parameters():
        p.requires_grad = False
    for patient_batch in loader:
        patient_batch = patient_batch.to(device)
        optimizer.zero_grad()
        logits_bb, _ = model(patient_batch)
        loss_bb = criterion_unweighted(logits_bb, patient_batch.patient_y)
        loss_bb.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Stage 2B: Extract deterministic bag embeddings and apply SMOTE in latent space
    model.eval()
    for p in model.classifier.parameters():
        p.requires_grad = True

    clean_embs_list, clean_targets_list = [], []
    with torch.no_grad():
        for patient_batch in loader:
            patient_batch = patient_batch.to(device)
            emb, _ = model.encode_patient_bag(patient_batch)
            clean_embs_list.append(emb)
            clean_targets_list.append(patient_batch.patient_y)

    clean_embs = torch.cat(clean_embs_list, dim=0)
    clean_targets = torch.cat(clean_targets_list, dim=0)
    embs_np, targets_np = clean_embs.cpu().numpy(), clean_targets.cpu().numpy()

    class_counts = np.bincount(targets_np)
    min_count = int(class_counts.min())
    if len(class_counts) > 1 and min_count >= 2:
        smote = SMOTE(k_neighbors=min(k_neighbors, min_count - 1), random_state=random_state + epoch)
        X_res_np, y_res_np = smote.fit_resample(embs_np, targets_np)
        combined_embs = torch.tensor(X_res_np, dtype=clean_embs.dtype, device=device)
        combined_targets = torch.tensor(y_res_np, dtype=torch.long, device=device)
    else:
        combined_embs, combined_targets = clean_embs, clean_targets

    model.classifier.train()
    perm = torch.randperm(combined_embs.size(0), device=device)
    clf_loss_sum, clf_samples = 0.0, 0
    for start in range(0, combined_embs.size(0), smote_batch_size):
        idx = perm[start : start + smote_batch_size]
        optimizer.zero_grad()
        batch_logits = model.classifier(combined_embs[idx])
        batch_loss = criterion_unweighted(batch_logits, combined_targets[idx])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
        optimizer.step()

        bs = idx.size(0)
        clf_loss_sum += batch_loss.item() * bs
        clf_samples += bs

    model.eval()
    with torch.no_grad():
        real_acc = (model.classifier(clean_embs).argmax(dim=1) == clean_targets).float().mean().item()

    return clf_loss_sum / max(1, clf_samples), real_acc


def save_latent_smote_visualizations(
    device, model, train_loader, label_encoder_patient, fold_num, output_dir, k_neighbors=3
):
    """Project training patient latent embeddings into 2D PCA before vs. after SMOTE."""
    from imblearn.over_sampling import SMOTE

    model.eval()
    embs_list, targets_list = [], []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            emb, _ = model.encode_patient_bag(batch)
            embs_list.append(emb.cpu().numpy())
            targets_list.append(batch.patient_y.cpu().numpy())

    X_train_emb = np.vstack(embs_list)
    y_train_emb = np.concatenate(targets_list)

    before_counts = Counter(y_train_emb)
    effective_k = max(1, min(k_neighbors, min(before_counts.values()) - 1))
    X_res, y_res = SMOTE(k_neighbors=effective_k, random_state=42).fit_resample(X_train_emb, y_train_emb)
    after_counts = Counter(y_res)

    class_names = [str(c) for c in label_encoder_patient.classes_]
    logging.info(
        f"[Fold {fold_num} Latent SMOTE] "
        f"Before: {{class_names[int(k)]: int(v) for k, v in sorted(before_counts.items())}} -> "
        f"After (k={effective_k}): {{class_names[int(k)]: int(v) for k, v in sorted(after_counts.items())}}"
    )

    pca = PCA(n_components=2, random_state=42)
    X_real_2d = pca.fit_transform(X_train_emb)
    X_res_2d = pca.transform(X_res)
    n_real = len(X_train_emb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1f77b4", "#d62728"]

    for cls_idx, cls_name in enumerate(class_names):
        color = colors[cls_idx % len(colors)]
        mask = y_train_emb == cls_idx
        ax1.scatter(X_real_2d[mask, 0], X_real_2d[mask, 1], c=color, label=f"{cls_name} (n={mask.sum()})", alpha=0.8, edgecolors="k", s=60)

        mask_real = y_res[:n_real] == cls_idx
        ax2.scatter(X_res_2d[:n_real][mask_real, 0], X_res_2d[:n_real][mask_real, 1], c=color, label=f"{cls_name} real (n={mask_real.sum()})", alpha=0.8, edgecolors="k", s=60)
        mask_syn = y_res[n_real:] == cls_idx
        if mask_syn.sum() > 0:
            ax2.scatter(X_res_2d[n_real:][mask_syn, 0], X_res_2d[n_real:][mask_syn, 1], c=color, marker="^", label=f"{cls_name} synthetic (n={mask_syn.sum()})", alpha=0.85, edgecolors="black", s=75)

    for ax, title in [(ax1, f"Fold {fold_num} - Before SMOTE"), (ax2, f"Fold {fold_num} - After Latent SMOTE (k={effective_k})")]:
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.legend()
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"latent_smote_pca_fold_{fold_num}.png"), dpi=300)
    plt.close()


def extract_and_save_patient_embeddings(
    device, model, loaders_dict, label_encoder_patient, fold_num, output_dir,
    clinical_file_path=None, clinical_features_to_include=None
):
    """Extract patient-level embeddings across splits, merge clinical annotations, and save TSV + 2D PCA."""
    model.eval()
    emb_dir = os.path.join(output_dir, "patient_embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    records = []
    with torch.no_grad():
        for split_name, loader in loaders_dict.items():
            for batch in loader:
                batch = batch.to(device)
                patient_emb, _ = model.encode_patient_bag(batch)
                probs = torch.softmax(model.classifier(patient_emb), dim=-1)
                targets = batch.patient_y
                raw_ids = getattr(batch, "patient_id", [f"Patient_{i}" for i in range(len(targets))])

                emb_np, probs_np = patient_emb.cpu().numpy(), probs.cpu().numpy()
                for idx in range(len(targets)):
                    y_true = int(targets[idx].item())
                    row = {
                        "patient_id": _unwrap_id(raw_ids[idx] if idx < len(raw_ids) else f"Patient_{idx}"),
                        "split": split_name,
                        "target_label": y_true,
                        "target_label_name": label_encoder_patient.classes_[y_true] if label_encoder_patient is not None else str(y_true),
                        "prob_resistant": round(float(probs_np[idx, 1]), 4) if probs_np.shape[1] > 1 else 0.0,
                    }
                    for d in range(emb_np.shape[1]):
                        row[f"emb_{d}"] = float(emb_np[idx, d])
                    records.append(row)

    df_emb = pd.DataFrame(records)
    if clinical_file_path is None:
        clinical_file_path = os.path.join(config_file_tree.PREPROCESSED_DATA_DIR, "preprocessed_clinical_data.tsv")
    if os.path.exists(clinical_file_path):
        df_clin = pd.read_csv(clinical_file_path, sep="\t")
        if "patient_id" in df_clin.columns:
            df_clin["patient_id"] = df_clin["patient_id"].astype(str).str.strip()
            if clinical_features_to_include is not None:
                valid_cols = ["patient_id"] + [c for c in clinical_features_to_include if c in df_clin.columns]
                df_clin = df_clin[valid_cols]
            df_emb = df_emb.merge(df_clin.drop_duplicates("patient_id"), on="patient_id", how="left")

    tsv_path = os.path.join(emb_dir, f"patient_embeddings_fold_{fold_num}.tsv")
    df_emb.to_csv(tsv_path, sep="\t", index=False)

    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    train_mask = (df_emb["split"] == "train").values
    if train_mask.sum() >= 3 and len(emb_cols) >= 2:
        pca = PCA(n_components=2, random_state=42).fit(df_emb.loc[train_mask, emb_cols].values)
        coords_2d = pca.transform(df_emb[emb_cols].values)
        df_emb["PC1"], df_emb["PC2"] = coords_2d[:, 0], coords_2d[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors, markers = {"no": "#1f77b4", "yes": "#d62728"}, {"train": "o", "val": "s", "test": "^"}
        for split_name in ["train", "val", "test"]:
            for cls_name in df_emb["target_label_name"].unique():
                sub = df_emb[(df_emb["split"] == split_name) & (df_emb["target_label_name"] == cls_name)]
                if not sub.empty:
                    ax.scatter(
                        sub["PC1"], sub["PC2"],
                        c=colors.get(cls_name, "#555555"), marker=markers.get(split_name, "o"),
                        s=70 if split_name != "test" else 95, alpha=0.85, edgecolors="k",
                        label=f"{cls_name} ({split_name}, n={len(sub)})",
                    )
        ax.set_title(f"Fold {fold_num} - Patient ABMIL Embeddings (PCA)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(emb_dir, f"patient_embeddings_pca_fold_{fold_num}.png"), dpi=300)
        plt.close()

    logging.info(f"[Fold {fold_num}] Saved patient embeddings to: {tsv_path}")
    return df_emb


# =============================================================================
# 4. FROZEN-GAT & LINEAR MIL (ELASTICNET / PLS-DA) UTILITIES
# =============================================================================
def load_gene_node_mapping(mapping_path=None):
    """Load gene_node_mapping.tsv sorted by node_index."""
    if mapping_path is None:
        mapping_path = os.path.join(config_file_tree.PROCESSED_SAMPLES_DIR, "gene_node_mapping.tsv")
    if not os.path.exists(mapping_path):
        logging.warning(f"gene_node_mapping.tsv not found at {mapping_path}. Falling back to node indices.")
        return None
    return pd.read_csv(mapping_path, sep="\t").sort_values("node_index").reset_index(drop=True)


def extract_frozen_gat_sample_embeddings(device, model_gat, loader):
    """Extract 64-dimensional sample embeddings from a frozen GAT encoder."""
    model_gat.eval()
    records = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, sample_embs = model_gat(
                batch.x, batch.edge_index, batch.batch, metadata_x=getattr(batch, "metadata_x", None)
            )
            embs_np = sample_embs.cpu().numpy()
            tissues_np = batch.y.cpu().numpy()
            sample_ids = list(batch.sample_id) if isinstance(batch.sample_id, (list, tuple)) else [batch.sample_id]
            patient_ids = list(batch.patient_id) if isinstance(batch.patient_id, (list, tuple)) else [batch.patient_id]

            for idx in range(embs_np.shape[0]):
                row = {
                    "sample_id": str(sample_ids[idx]).strip(),
                    "patient_id": str(patient_ids[idx]).strip(),
                    "tissue_label": int(tissues_np[idx]),
                }
                for d in range(embs_np.shape[1]):
                    row[f"gat_emb_{d}"] = float(embs_np[idx, d])
                records.append(row)

    return pd.DataFrame(records)


def deconfound_tissue_embeddings(df_train, df_val, df_test, emb_cols, tissue_col="tissue_label", scale_by_tissue=True):
    """
    Remove tissue-type confounding using statistics computed strictly on the training split,
    followed by per-sample L2 normalization.
    """
    global_mean = df_train[emb_cols].mean(axis=0).values
    global_std = np.clip(df_train[emb_cols].std(axis=0).values, 1e-6, None)
    tissue_means = df_train.groupby(tissue_col)[emb_cols].mean()
    tissue_stds = df_train.groupby(tissue_col)[emb_cols].std().fillna(1.0)
    tissue_counts = df_train.groupby(tissue_col).size()

    def _transform_df(df_split):
        df_out = df_split.copy()
        X = df_out[emb_cols].values.copy()
        for idx, t_val in enumerate(df_out[tissue_col].values):
            if t_val in tissue_means.index:
                X[idx] -= tissue_means.loc[t_val].values
                if scale_by_tissue:
                    std_vec = np.clip(tissue_stds.loc[t_val].values, 1e-6, None) if tissue_counts.loc[t_val] >= 3 else global_std
                    X[idx] /= std_vec
            else:
                X[idx] -= global_mean
                if scale_by_tissue:
                    X[idx] /= global_std

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        df_out[emb_cols] = X / np.clip(norms, 1e-6, None)
        return df_out

    return _transform_df(df_train), _transform_df(df_val), _transform_df(df_test)


def _compute_bag_pool_weights(X_bag, pooling_mode="softmax", tau=0.20):
    """
    Compute per-sample pooling weight matrix W [n_samples, 64] for a patient bag:
      - 'softmax': temperature-scaled smooth-max across biopsies
      - 'top2': equal weight on top-2 biopsies per latent dimension
      - 'max': hard argmax per latent dimension
    """
    n_samples, n_dims = X_bag.shape
    if n_samples == 1 or pooling_mode == "mean":
        W = np.full((n_samples, n_dims), 1.0 / n_samples, dtype=np.float64)
    elif pooling_mode == "softmax":
        scaled = X_bag / max(float(tau), 1e-6)
        scaled -= scaled.max(axis=0, keepdims=True)
        exp_x = np.exp(scaled)
        W = exp_x / np.clip(exp_x.sum(axis=0, keepdims=True), 1e-12, None)
    elif pooling_mode == "top2":
        k = min(2, n_samples)
        W = np.zeros((n_samples, n_dims), dtype=np.float64)
        top_idx = np.argpartition(-X_bag, kth=k - 1, axis=0)[:k, :]
        W[top_idx, np.arange(n_dims)[None, :]] = 1.0 / k
    else:
        W = np.zeros((n_samples, n_dims), dtype=np.float64)
        W[X_bag.argmax(axis=0), np.arange(n_dims)] = 1.0

    return np.sum(W * X_bag, axis=0), W


def aggregate_patient_bags(
    df_samples, emb_cols, patient_target_map, label_encoder_sample=None, pooling_mode="softmax", tau=0.20
):
    """
    Aggregate biopsy embeddings per patient into a 128-D vector [Mean || Pooled]
    and record per-sample pooling weights for exact gradient backpropagation.
    """
    records, pool_weights_map = [], {}

    for patient_id, grp in df_samples.groupby("patient_id"):
        if patient_id not in patient_target_map:
            continue

        X_bag = grp[emb_cols].values.astype(np.float64)
        s_ids = [str(s).strip() for s in grp["sample_id"].values]
        mean_vec = X_bag.mean(axis=0)
        pooled_vec, W = _compute_bag_pool_weights(X_bag, pooling_mode=pooling_mode, tau=tau)

        pool_weights_map[patient_id] = {s_ids[j]: W[j] for j in range(len(s_ids))}
        tissues = sorted(
            {label_encoder_sample.classes_[t] if label_encoder_sample is not None else str(t) for t in grp["tissue_label"].values}
        )

        row = {
            "patient_id": patient_id,
            "target_label": int(patient_target_map[patient_id]),
            "n_samples": len(grp),
            "tissue_types": ", ".join(tissues),
        }
        for d, val in enumerate(mean_vec):
            row[f"mean_emb_{d}"] = float(val)
        for d, val in enumerate(pooled_vec):
            row[f"max_emb_{d}"] = float(val)
        records.append(row)

    return pd.DataFrame(records), pool_weights_map


def compute_plsda_vip_scores(pls_model):
    """Compute Variable Importance in Projection (VIP) scores from a fitted PLS model."""
    t, w, q = pls_model.x_scores_, pls_model.x_weights_, pls_model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    if total_s <= 0:
        return np.zeros(p)
    w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (w_norm ** 2 @ s).ravel() / total_s)


def compute_gat_gene_attribution_exact_bag(
    device, model_gat, graphs_list, max_sample_map, latent_weights_128,
    df_gene_mapping=None, feature_names=("gene_expression", "log2CopyNumber"),
):
    """
    Backpropagate bag-level linear weights [w_mean || w_pool] through the frozen GAT
    encoder onto input gene features and map node indices to HGNC symbols.
    """
    model_gat.eval()
    w_mean, w_max = latent_weights_128[:64], latent_weights_128[64:]

    patient_graphs = {}
    for g in graphs_list:
        pid = str(g.patient_id).strip()
        if pid in max_sample_map:
            patient_graphs.setdefault(pid, []).append(g)

    grad_sum_signed, grad_sum_abs, n_patients_used = None, None, 0
    for pid, p_graphs in patient_graphs.items():
        n_bag = float(len(p_graphs))
        sample_w_dict = max_sample_map[pid]

        for g in p_graphs:
            sid = str(g.sample_id).strip()
            v_sample_np = (w_mean / n_bag) + (w_max * sample_w_dict.get(sid, np.zeros(64, dtype=np.float64)))
            if np.allclose(v_sample_np, 0.0):
                continue

            v_sample = torch.tensor(v_sample_np, dtype=torch.float32, device=device)
            g_dev = g.to(device)
            x_in = g_dev.x.detach().clone().requires_grad_(True)
            batch_vec = torch.zeros(x_in.size(0), dtype=torch.long, device=device)

            _, sample_emb = model_gat(x_in, g_dev.edge_index, batch_vec, metadata_x=getattr(g_dev, "metadata_x", None))
            grad_x = torch.autograd.grad(torch.sum(sample_emb.squeeze(0) * v_sample), x_in)[0].detach().cpu().numpy()

            if grad_sum_signed is None:
                grad_sum_signed = np.zeros_like(grad_x, dtype=np.float64)
                grad_sum_abs = np.zeros_like(grad_x, dtype=np.float64)
            grad_sum_signed += grad_x
            grad_sum_abs += np.abs(grad_x)

        n_patients_used += 1

    mean_signed = grad_sum_signed / max(1, n_patients_used)
    mean_abs = grad_sum_abs / max(1, n_patients_used)
    num_genes = mean_abs.shape[0]

    if df_gene_mapping is not None and len(df_gene_mapping) == num_genes:
        gene_ids = df_gene_mapping["gene_id"].astype(str).tolist()
        gene_symbols = df_gene_mapping["gene_symbol"].astype(str).tolist()
    else:
        gene_ids = [f"gene_{i}" for i in range(num_genes)]
        gene_symbols = [f"gene_{i}" for i in range(num_genes)]

    records = []
    for g_idx in range(num_genes):
        row = {
            "node_index": g_idx,
            "gene_id": gene_ids[g_idx],
            "gene_symbol": gene_symbols[g_idx],
            "total_abs_importance": float(mean_abs[g_idx].sum()),
        }
        for f_idx, f_name in enumerate(feature_names):
            row[f"{f_name}_abs_grad"] = float(mean_abs[g_idx, f_idx])
            row[f"{f_name}_signed_grad"] = float(mean_signed[g_idx, f_idx])
        records.append(row)

    return pd.DataFrame(records).sort_values("total_abs_importance", ascending=False)


# =============================================================================
# 5. REPORTING, LEARNING CURVES & SUBGROUP EVALUATION
# =============================================================================
def show_fold_results(df_fold_results, task, save_to_csv=False, dir_to_save=None, filename=None):
    """Log Mean +/- Variance across folds and optionally save the fold metrics table."""
    ignore_cols = {"fold", "total_samples", "total_patients"}
    metric_cols = [c for c in df_fold_results.columns if c not in ignore_cols and not c.startswith("n_")]

    mean_series, var_series = df_fold_results[metric_cols].mean(), df_fold_results[metric_cols].var()
    logging.info("--- OVERALL FOLD STATISTICS (MEAN ± VAR) ---")
    for col in metric_cols:
        logging.info(f"{col}: {mean_series[col]:.4f} ± {var_series[col]:.4f} (Var)")

    if save_to_csv and dir_to_save:
        out_path = os.path.join(dir_to_save, filename or f"report_kfold_test_results_{task}.csv")
        df_fold_results.to_csv(out_path, index=False)
        logging.info(f"Final folds report saved to: {out_path}\n")


def save_learning_curves(history, fold_num, output_dir="learning_curves", is_grid_search=False):
    """Plot and save training/validation loss and accuracy trajectories."""
    epochs = range(1, len(history["train_loss"]) + 1)
    prefix = f"Cfg {fold_num}" if is_grid_search else f"Fold {fold_num}"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", linestyle="--", linewidth=2)
    ax1.set_title(f"{prefix} - Loss Trajectory")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if any(m > 0 for m in history["train_metric"]):
        ax2.plot(epochs, history["train_metric"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history["val_metric"], label="Val Accuracy", color="#ff7f0e", linestyle="--", linewidth=2)
    ax2.set_title(f"{prefix} - Accuracy Performance")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"cfg{fold_num}.png" if is_grid_search else f"learning_curve_fold{fold_num}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()


def set_sample_encoder_freezing(model, freeze: bool = True):
    """Freeze or unfreeze parameters of the MIL sample encoder (GAT)."""
    for param in model.sample_encoder.parameters():
        param.requires_grad = not freeze
    if freeze:
        model.sample_encoder.eval()
        logging.info("Sample Encoder (GAT) frozen. Training MIL head only.\n")
    else:
        model.sample_encoder.train()
        logging.info("Sample Encoder (GAT) unfrozen. Fine-tuning enabled.\n")


def apply_permutation_test(patient_dataset, seed=42):
    """Randomly permute patient target labels across bags as a negative control."""
    rng = np.random.default_rng(seed)
    original_labels = np.array([bag.patient_y.item() for bag in patient_dataset])
    permuted_labels = rng.permutation(original_labels)

    for idx, bag in enumerate(patient_dataset):
        bag.patient_y = torch.tensor(permuted_labels[idx], dtype=torch.long)

    logging.info(
        f"[PERMUTATION TEST] Labels shuffled. Original: {np.bincount(original_labels)} | "
        f"Permuted: {np.bincount(permuted_labels)}"
    )
    return patient_dataset


def save_patient_classification_results(
    device, model, test_loader, label_encoder_patient, fold_num, output_dir,
    label_encoder_sample=None, clinical_file_path=None, clinical_features_to_include=None
):
    """
    Evaluate a neural MIL model on test patients, merge clinical covariates, and save
    separate TSV reports for correct and misclassified patients.
    """
    model.eval()
    predictions_dir = os.path.join(output_dir, "patient_predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    records = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            targets = batch.patient_y if hasattr(batch, "patient_y") else batch.y
            raw_ids = getattr(batch, "patient_id", [f"Patient_{i}" for i in range(len(targets))])

            if hasattr(batch, "y") and batch.y is not None:
                y_tissues = batch.y.cpu().numpy()
                batch_tissues = [label_encoder_sample.classes_[t] if label_encoder_sample is not None else str(t) for t in y_tissues]
            elif hasattr(batch, "tissue_type"):
                batch_tissues = [str(t) for t in batch.tissue_type]
            else:
                batch_tissues = []

            ns_per_pat = (
                batch.num_samples if hasattr(batch, "num_samples") and batch.num_samples is not None
                else ([len(batch_tissues)] if len(targets) == 1 else [])
            )

            offset = 0
            for idx in range(len(targets)):
                y_true, y_pred = targets[idx].item(), preds[idx].item()
                p_0 = probs[idx][0].item()
                p_1 = probs[idx][1].item() if probs.shape[1] > 1 else (1.0 - p_0)

                if offset < len(batch_tissues) and idx < len(ns_per_pat):
                    n_s = ns_per_pat[idx]
                    tissues_str = ", ".join(sorted(set(batch_tissues[offset : offset + n_s])))
                    offset += n_s
                else:
                    tissues_str = "Unknown"

                records.append({
                    "patient_id": _unwrap_id(raw_ids[idx] if idx < len(raw_ids) else f"Patient_{idx}"),
                    "target_label": y_true,
                    "target_label_name": label_encoder_patient.classes_[y_true] if label_encoder_patient is not None else str(y_true),
                    "predicted_label": y_pred,
                    "predicted_label_name": label_encoder_patient.classes_[y_pred] if label_encoder_patient is not None else str(y_pred),
                    "prob_class_0": round(p_0, 4),
                    "prob_class_1": round(p_1, 4),
                    "tissue_types": tissues_str,
                    "is_correct": (y_true == y_pred),
                })

    df_results = pd.DataFrame(records)
    df_results["patient_id"] = df_results["patient_id"].astype(str)

    if clinical_file_path is None:
        clinical_file_path = os.path.join(config_file_tree.PREPROCESSED_DATA_DIR, "preprocessed_clinical_data.tsv")
    if os.path.exists(clinical_file_path):
        df_clin = pd.read_csv(clinical_file_path, sep="\t")
        if "patient_id" in df_clin.columns:
            df_clin["patient_id"] = df_clin["patient_id"].astype(str).str.strip()
            if clinical_features_to_include is not None:
                valid_cols = ["patient_id"] + [c for c in clinical_features_to_include if c in df_clin.columns]
                df_clin = df_clin[valid_cols]
            df_results = df_results.merge(df_clin.drop_duplicates("patient_id"), on="patient_id", how="left")

    subgroup_metrics = evaluate_subgroups(df_results, subgroup_col="Treatment_strategy")

    df_correct = df_results[df_results["is_correct"]].drop(columns=["is_correct"])
    df_misclassified = df_results[~df_results["is_correct"]].drop(columns=["is_correct"])

    df_correct.to_csv(os.path.join(predictions_dir, f"fold_{fold_num}_correct_patients.tsv"), sep="\t", index=False)
    df_misclassified.to_csv(os.path.join(predictions_dir, f"fold_{fold_num}_misclassified_patients.tsv"), sep="\t", index=False)
    logging.info(f"[Fold {fold_num}] Saved predictions -> Correct: {len(df_correct)} | Misclassified: {len(df_misclassified)}")
    return df_correct, df_misclassified, subgroup_metrics


def evaluate_subgroups(df_results, subgroup_col="Treatment_strategy"):
    """Evaluate classification performance (Accuracy, F1, ROC-AUC) per clinical subgroup."""
    if subgroup_col not in df_results.columns:
        logging.warning(f"[SUBGROUP ANALYSIS] Column '{subgroup_col}' not found in results DataFrame.")
        return {}

    logging.info(f"--- SUBGROUP ANALYSIS ({subgroup_col}) ---")
    subgroup_metrics = {}
    for group_val, df_grp in df_results.dropna(subset=[subgroup_col]).groupby(subgroup_col):
        y_true, y_pred = df_grp["target_label"].values, df_grp["predicted_label"].values
        n_classes = len(np.unique(y_true))
        auc_val = np.nan

        try:
            if n_classes == 2 and "prob_class_1" in df_grp.columns:
                auc_val = roc_auc_score(y_true, df_grp["prob_class_1"].values)
            elif n_classes > 2:
                prob_cols = [c for c in df_grp.columns if c.startswith("prob_class_")]
                if len(prob_cols) > 1:
                    auc_val = roc_auc_score(y_true, df_grp[prob_cols].values, multi_class="ovr")
        except ValueError:
            auc_val = np.nan

        metrics = {
            "subgroup": group_val,
            "n_patients": len(df_grp),
            "acc": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "auc": auc_val,
        }
        subgroup_metrics[group_val] = metrics
        auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "N/A (single class)"
        logging.info(f"  > Subgroup [{group_val}] (N={len(df_grp)}) | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {auc_str}")

    return subgroup_metrics