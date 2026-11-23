import logging
import os
from collections import defaultdict
from datetime import datetime

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config_file_tree
from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata
from utils_train_eval import (
    handle_pretrained_gat_checkpoint,
    pre_train_imputation_encoding,
    print_details,
    save_training_config_report,
)
from train_functions import (
    SelectiveNodeScaler,
    evaluate,
    evaluate_subgroups,
    save_learning_curves,
    show_fold_results,
    train_epoch,
)

# Import flessibile del modello salvato in models/PatientMLP.py
try:
    from models.PatientMLP import PatientBottleneckMLP as PatientMLP
except ImportError:
    from models.PatientMLP import PatientMLP


# ==============================================================================
# CONFIGURAZIONE ESPERIMENTO MLP PATIENT-LEVEL (VARIANTI A & B)
# ==============================================================================
# Puoi testare una singola variante (es. ["A"] o ["B"]) oppure entrambe in sequenza:
VARIANTS_TO_RUN = ["A", "B"]

# Opzioni Variante A (Aggregazione embedding GAT congelati):
DECONFOUND_TISSUE_IN_VAR_A = True   # Sottrae la media per tissue_type stimata sul solo Train set

# Opzioni Variante B (Aggregazione feature omiche grezze per paziente):
TOP_K_OMIC_FEATURES = 300           # Numero di feature omiche selezionate (f_classif) sul solo Train set

# Iperparametri MLP compatto (adatti a coorti piccole es. NACT N=78):
MLP_BOTTLENECK_DIM = 16
MLP_DROPOUT = 0.4
MLP_EPOCHS = 120
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 5e-2
MLP_EARLY_STOPPING_PATIENCE = 20
EMA_ALPHA = 0.3                     # Smussamento Val Loss su validation set piccoli
# ==============================================================================


# ==============================================================================
# FUNZIONI DI AGGREGAZIONE TESSUTI A LIVELLO PAZIENTE (VARIANTI A & B)
# ==============================================================================

@torch.no_grad()
def extract_sample_gat_embeddings(device, gat_model, samples_list):
    """
    Estrae gli embedding a livello di singolo campione/biopsia dalla GAT congelata.
    Restituisce un dizionario sample_id -> np.ndarray [emb_dim].
    """
    gat_model.eval()
    loader = DataLoader(samples_list, batch_size=32, shuffle=False, num_workers=0)
    sample_emb_map = {}

    idx_offset = 0
    for batch in loader:
        batch = batch.to(device)
        metadata_x = getattr(batch, 'metadata_x', None)
        _, embs = gat_model(batch.x, batch.edge_index, batch.batch, metadata_x=metadata_x)
        embs_np = embs.cpu().numpy()

        for i in range(embs_np.shape[0]):
            g = samples_list[idx_offset + i]
            sid = str(getattr(g, 'sample_id', f'sample_{idx_offset + i}')).strip()
            sample_emb_map[sid] = embs_np[i]
        idx_offset += embs_np.shape[0]

    return sample_emb_map


def build_patient_vectors_variant_a(
    samples_list,
    patient_target_map,
    sample_emb_map,
    num_tissue_classes=3,
    tissue_class_names=None,
    patient_clinical_map=None,
    clinical_feature_names=None,
    tissue_means_train=None,
    fit_tissue_means=False,
):
    """
    VARIANTE A: Aggrega per ciascun paziente gli embedding della GAT congelata:
      [Mean_emb (D) || Max_emb (D) || Tissue_proportions (C) || Clinical_features (optional)]
    Se fit_tissue_means=True (solo sul Train set), calcola i centroidi per tissue_type
    per rimuovere il confondente anatomico prima dell'aggregazione.
    """
    if tissue_class_names is None:
        tissue_class_names = [f"tissue_{i}" for i in range(num_tissue_classes)]

    # 1. Stima o applicazione del de-confounding tissutale
    if fit_tissue_means:
        tissue_vecs = defaultdict(list)
        for g in samples_list:
            sid = str(g.sample_id).strip()
            t_lbl = int(g.y.item())
            tissue_vecs[t_lbl].append(sample_emb_map[sid])
        tissue_means_train = {
            t: np.mean(vecs, axis=0) for t, vecs in tissue_vecs.items() if len(vecs) > 0
        }

    # 2. Raggruppa i campioni per patient_id
    patient_graphs = defaultdict(list)
    for g in samples_list:
        pid = str(g.patient_id).strip()
        if pid in patient_target_map:
            patient_graphs[pid].append(g)

    patient_ids = sorted(patient_graphs.keys())
    X_rows, y_rows, dominant_tissues = [], [], []

    emb_dim = next(iter(sample_emb_map.values())).shape[0]
    feature_names = (
        [f"GAT_mean_{i}" for i in range(emb_dim)]
        + [f"GAT_max_{i}" for i in range(emb_dim)]
        + [f"prop_{name}" for name in tissue_class_names]
    )
    if patient_clinical_map and clinical_feature_names:
        feature_names += list(clinical_feature_names)

    for pid in patient_ids:
        graphs = patient_graphs[pid]
        embs = []
        tissues = []
        for g in graphs:
            sid = str(g.sample_id).strip()
            t_lbl = int(g.y.item())
            vec = sample_emb_map[sid].copy()
            if tissue_means_train is not None and t_lbl in tissue_means_train:
                vec = vec - tissue_means_train[t_lbl]
            embs.append(vec)
            tissues.append(t_lbl)

        embs_arr = np.vstack(embs)  # [N_samples_pid, emb_dim]
        mean_emb = embs_arr.mean(axis=0)
        max_emb = embs_arr.max(axis=0)

        tissues_arr = np.array(tissues, dtype=int)
        tissue_props = np.bincount(tissues_arr, minlength=num_tissue_classes).astype(np.float32) / len(graphs)
        dom_idx = int(np.argmax(tissue_props))
        dominant_tissues.append(tissue_class_names[dom_idx])

        parts = [mean_emb, max_emb, tissue_props]
        if patient_clinical_map and pid in patient_clinical_map:
            parts.append(patient_clinical_map[pid].cpu().numpy())

        X_rows.append(np.concatenate(parts, axis=0).astype(np.float32))
        y_rows.append(int(patient_target_map[pid]))

    return (
        np.vstack(X_rows),
        np.array(y_rows, dtype=np.int64),
        patient_ids,
        feature_names,
        dominant_tissues,
        tissue_means_train,
    )


def build_patient_vectors_variant_b(
    samples_list,
    patient_target_map,
    active_feature_names,
    num_tissue_classes=3,
    tissue_class_names=None,
    patient_clinical_map=None,
    clinical_feature_names=None,
    selector=None,
    top_k=300,
    fit_selector=False,
):
    """
    VARIANTE B: Aggrega direttamente le feature omiche scalate dei nodi tra i campioni
    di ciascun paziente (Mean sui tessuti del paziente) + selezione univariata (f_classif)
    eseguita ESCLUSIVAMENTE sul training set del fold per prevenire data leakage.
    """
    if tissue_class_names is None:
        tissue_class_names = [f"tissue_{i}" for i in range(num_tissue_classes)]

    patient_graphs = defaultdict(list)
    for g in samples_list:
        pid = str(g.patient_id).strip()
        if pid in patient_target_map:
            patient_graphs[pid].append(g)

    patient_ids = sorted(patient_graphs.keys())
    num_nodes, num_feats = samples_list[0].x.shape

    raw_omic_names = [
        f"gene_{g_idx:04d}_{feat_name}"
        for g_idx in range(num_nodes)
        for feat_name in active_feature_names
    ]

    X_omic_rows, y_rows, tissue_props_rows, clin_rows, dominant_tissues = [], [], [], [], []

    for pid in patient_ids:
        graphs = patient_graphs[pid]
        # Media delle matrici [num_nodes, num_feats] tra tutte le biopsie del paziente
        stacked_x = np.stack([g.x.cpu().numpy() for g in graphs], axis=0)  # [N_samp, 7191, F]
        mean_omic_flat = stacked_x.mean(axis=0).reshape(-1).astype(np.float32)

        tissues_arr = np.array([int(g.y.item()) for g in graphs], dtype=int)
        tissue_props = np.bincount(tissues_arr, minlength=num_tissue_classes).astype(np.float32) / len(graphs)
        dom_idx = int(np.argmax(tissue_props))
        dominant_tissues.append(tissue_class_names[dom_idx])

        X_omic_rows.append(mean_omic_flat)
        tissue_props_rows.append(tissue_props)
        y_rows.append(int(patient_target_map[pid]))

        if patient_clinical_map and pid in patient_clinical_map:
            clin_rows.append(patient_clinical_map[pid].cpu().numpy())

    X_omic = np.vstack(X_omic_rows)
    y_arr = np.array(y_rows, dtype=np.int64)
    tissue_props_mat = np.vstack(tissue_props_rows)

    # Selezione Top-K feature omiche (fit solo su Train!)
    k_eff = min(top_k, X_omic.shape[1])
    if fit_selector:
        # Rimuove eventuali colonne a varianza zero prima di f_classif
        selector = SelectKBest(score_func=f_classif, k=k_eff)
        X_omic_sel = selector.fit_transform(X_omic, y_arr)
    else:
        if selector is None:
            raise ValueError("selector deve essere fornito quando fit_selector=False")
        X_omic_sel = selector.transform(X_omic)

    selected_mask = selector.get_support()
    selected_omic_names = [name for name, keep in zip(raw_omic_names, selected_mask) if keep]

    parts = [X_omic_sel, tissue_props_mat]
    feature_names = selected_omic_names + [f"prop_{name}" for name in tissue_class_names]

    if clin_rows:
        parts.append(np.vstack(clin_rows))
        feature_names += list(clinical_feature_names)

    X_full = np.hstack(parts).astype(np.float32)
    return X_full, y_arr, patient_ids, feature_names, dominant_tissues, selector


# ==============================================================================
# TRAINING & EVALUATION HELPERS PER PatientMLP
# ==============================================================================

def _forward_mlp(model, x_tensor):
    """Gestisce sia forward -> (logits, z) sia forward -> logits + model.encode(x)."""
    out = model(x_tensor)
    if isinstance(out, tuple):
        return out[0], out[1]
    if hasattr(model, 'encode'):
        return out, model.encode(x_tensor)
    return out, out


def compute_classification_metrics(y_true, y_pred, y_probs):
    num_classes = len(np.unique(y_true))
    if num_classes == 2:
        auc_val = roc_auc_score(y_true, y_probs[:, 1])
    elif num_classes > 2:
        auc_val = roc_auc_score(y_true, y_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auc': auc_val,
    }


def save_mlp_patient_predictions(
    patient_ids,
    y_true,
    probs,
    dominant_tissues,
    label_encoder_patient,
    fold_num,
    output_dir,
    clinical_features_to_include=None,
):
    """
    Salva i TSV di predizioni corrette/errate per fold e calcola le metriche per sottogruppo.
    """
    predictions_dir = os.path.join(output_dir, "patient_predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    preds = np.argmax(probs, axis=1)
    records = []
    for idx, pid in enumerate(patient_ids):
        yt = int(y_true[idx])
        yp = int(preds[idx])
        p0 = float(probs[idx, 0])
        p1 = float(probs[idx, 1]) if probs.shape[1] > 1 else (1.0 - p0)
        records.append({
            'patient_id': str(pid).strip(),
            'target_label': yt,
            'target_label_name': str(label_encoder_patient.classes_[yt]),
            'predicted_label': yp,
            'predicted_label_name': str(label_encoder_patient.classes_[yp]),
            'prob_class_0': round(p0, 4),
            'prob_class_1': round(p1, 4),
            'dominant_tissue': dominant_tissues[idx],
            'is_correct': (yt == yp),
        })

    df_results = pd.DataFrame(records)

    clinical_file_path = os.path.join(
        config_file_tree.PREPROCESSED_DATA_DIR,
        "preprocessed_clinical_data.tsv",
    )
    if os.path.exists(clinical_file_path):
        df_clin = pd.read_csv(clinical_file_path, sep='\t')
        if 'patient_id' in df_clin.columns:
            df_clin['patient_id'] = df_clin['patient_id'].astype(str).str.strip()
            if clinical_features_to_include:
                valid_cols = ['patient_id'] + [c for c in clinical_features_to_include if c in df_clin.columns]
                df_clin = df_clin[valid_cols].drop_duplicates(subset=['patient_id'])
            else:
                df_clin = df_clin.drop_duplicates(subset=['patient_id'])
            df_results = df_results.merge(df_clin, on='patient_id', how='left')

    subgroup_metrics = evaluate_subgroups(df_results, subgroup_col='Treatment_strategy')

    df_correct = df_results[df_results['is_correct']].drop(columns=['is_correct'])
    df_misclassified = df_results[~df_results['is_correct']].drop(columns=['is_correct'])

    df_correct.to_csv(os.path.join(predictions_dir, f"fold_{fold_num}_correct_patients.tsv"), sep='\t', index=False)
    df_misclassified.to_csv(os.path.join(predictions_dir, f"fold_{fold_num}_misclassified_patients.tsv"), sep='\t', index=False)

    return df_results, subgroup_metrics


# ==============================================================================
# ESTRAZIONE INSIGHTS DAGLI EMBEDDING (PCA / UMAP + COSINE SIMILARITY + PESI)
# ==============================================================================

def extract_and_save_embedding_insights(
    oof_records,
    feature_importance_records,
    label_encoder_patient,
    variant_name,
    output_dir,
):
    """
    Genera:
      1. Tabella TSV con gli embedding Out-Of-Fold (senza leakage) + metadati clinici.
      2. Proiezioni 2D PCA e UMAP (con fallback t-SNE) colorate per resistenza, probabilità,
         tessuto dominante ed eventuale stato HRD.
      3. Matrice di similarità coseno ordinata per classe e clustering gerarchico, con test
         statistico Mann-Whitney sulla coesione intra-classe vs inter-classe.
      4. Ranking delle feature con maggiore peso effettivo nel modello lineare/bottleneck.
    """
    insights_dir = os.path.join(output_dir, "embedding_insights")
    os.makedirs(insights_dir, exist_ok=True)

    df_oof = pd.DataFrame(oof_records)
    emb_cols = [c for c in df_oof.columns if c.startswith("emb_")]
    Z = df_oof[emb_cols].values.astype(np.float32)

    # Merge con eventuali variabili cliniche extra per interpretazione biologica
    clinical_file_path = os.path.join(
        config_file_tree.PREPROCESSED_DATA_DIR,
        "preprocessed_clinical_data.tsv",
    )
    extra_clin_cols = ['Treatment_strategy', 'is_hrd', 'Residual_tumor', 'Stage_FIGO2014', 'PFI_KaplanM_allHGSC']
    if os.path.exists(clinical_file_path):
        df_clin = pd.read_csv(clinical_file_path, sep='\t')
        if 'patient_id' in df_clin.columns:
            df_clin['patient_id'] = df_clin['patient_id'].astype(str).str.strip()
            avail_cols = ['patient_id'] + [c for c in extra_clin_cols if c in df_clin.columns]
            df_oof = df_oof.merge(df_clin[avail_cols].drop_duplicates(subset=['patient_id']), on='patient_id', how='left')

    oof_tsv_path = os.path.join(insights_dir, f"oof_patient_embeddings_variant_{variant_name}.tsv")
    df_oof.to_csv(oof_tsv_path, sep='\t', index=False)
    logging.info(f"[INSIGHTS Var {variant_name}] Salvati embedding OOF ({len(df_oof)} pazienti) in: {oof_tsv_path}")

    # ------------------------------------------------------------------
    # 1. PROIEZIONI 2D: PCA & UMAP (o t-SNE fallback)
    # ------------------------------------------------------------------
    n_samples = Z.shape[0]
    if n_samples >= 5:
        pca = PCA(n_components=2, random_state=42)
        z_pca = pca.fit_transform(Z)
        var_exp = pca.explained_variance_ratio_ * 100

        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=min(10, max(3, n_samples // 5)),
                min_dist=0.25,
                metric='cosine',
                random_state=42,
            )
            z_nonlin = reducer.fit_transform(Z)
            nonlin_name = "UMAP (cosine)"
        except ImportError:
            perp = min(15, max(3, (n_samples - 1) // 3))
            reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
            z_nonlin = reducer.fit_transform(Z)
            nonlin_name = f"t-SNE (perp={perp})"

        df_oof['PC1'], df_oof['PC2'] = z_pca[:, 0], z_pca[:, 1]
        df_oof['NL1'], df_oof['NL2'] = z_nonlin[:, 0], z_nonlin[:, 1]
        df_oof.to_csv(oof_tsv_path, sep='\t', index=False)

        has_hrd = ('is_hrd' in df_oof.columns) and df_oof['is_hrd'].notna().any()
        n_rows_fig = 2
        fig, axes = plt.subplots(n_rows_fig, 2, figsize=(14, 11))

        # Pannello (0,0): PCA colorato per classe reale
        ax = axes[0, 0]
        palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
        for cls_idx, cls_name in enumerate(label_encoder_patient.classes_):
            mask = df_oof['target_label'] == cls_idx
            ax.scatter(
                df_oof.loc[mask, 'PC1'], df_oof.loc[mask, 'PC2'],
                label=f"{cls_name} (n={mask.sum()})",
                color=palette[cls_idx % len(palette)],
                s=75, alpha=0.88, edgecolors='k', linewidth=0.6,
            )
        ax.set_title(f"Var {variant_name} - PCA by Platinum Resistance\n(PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)

        # Pannello (0,1): UMAP/t-SNE colorato per classe reale
        ax = axes[0, 1]
        for cls_idx, cls_name in enumerate(label_encoder_patient.classes_):
            mask = df_oof['target_label'] == cls_idx
            ax.scatter(
                df_oof.loc[mask, 'NL1'], df_oof.loc[mask, 'NL2'],
                label=f"{cls_name} (n={mask.sum()})",
                color=palette[cls_idx % len(palette)],
                s=75, alpha=0.88, edgecolors='k', linewidth=0.6,
            )
        ax.set_title(f"Var {variant_name} - {nonlin_name} by Platinum Resistance")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)

        # Pannello (1,0): UMAP/t-SNE colorato per probabilità predetta di resistenza
        ax = axes[1, 0]
        sc = ax.scatter(
            df_oof['NL1'], df_oof['NL2'],
            c=df_oof['prob_class_1'], cmap='coolwarm',
            vmin=0.0, vmax=1.0, s=75, alpha=0.9, edgecolors='k', linewidth=0.6,
        )
        plt.colorbar(sc, ax=ax, label="Predicted P(Resistant)")
        ax.set_title(f"Var {variant_name} - {nonlin_name} by Predicted Resistance Prob")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.25)

        # Pannello (1,1): UMAP/t-SNE colorato per tessuto dominante (o stato HRD se disponibile)
        ax = axes[1, 1]
        strat_col = 'is_hrd' if has_hrd else 'dominant_tissue'
        strat_title = "HRD Status" if has_hrd else "Dominant Tissue Type"
        sub_vals = df_oof[strat_col].fillna("Missing").astype(str)
        for idx_v, val_name in enumerate(sorted(sub_vals.unique())):
            mask = sub_vals == val_name
            ax.scatter(
                df_oof.loc[mask, 'NL1'], df_oof.loc[mask, 'NL2'],
                label=f"{val_name} (n={mask.sum()})",
                color=palette[idx_v % len(palette)],
                s=75, alpha=0.88, edgecolors='k', linewidth=0.6,
            )
        ax.set_title(f"Var {variant_name} - {nonlin_name} by {strat_title}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        proj_fig_path = os.path.join(insights_dir, f"embedding_pca_umap_variant_{variant_name}.png")
        plt.savefig(proj_fig_path, dpi=300)
        plt.close()
        logging.info(f"[INSIGHTS Var {variant_name}] Salvati plot PCA/UMAP in: {proj_fig_path}")

    # ------------------------------------------------------------------
    # 2. MATRICE DI SIMILARITÀ COSENO ORDINATA PER CLASSE & CLUSTERING
    # ------------------------------------------------------------------
    if n_samples >= 4:
        ordered_indices = []
        class_boundaries = []
        running_count = 0

        for cls_idx in sorted(df_oof['target_label'].unique()):
            cls_indices = np.where(df_oof['target_label'].values == cls_idx)[0]
            if len(cls_indices) > 2:
                Z_cls = Z[cls_indices]
                # Clustering gerarchico interno alla classe per far emergere sottogruppi
                dists = pdist(Z_cls, metric='cosine')
                dists = np.nan_to_num(dists, nan=1.0)
                link = linkage(dists, method='average')
                order_in_cls = leaves_list(link)
                ordered_indices.extend(cls_indices[order_in_cls].tolist())
            else:
                ordered_indices.extend(cls_indices.tolist())
            running_count += len(cls_indices)
            class_boundaries.append(running_count)

        Z_ord = Z[ordered_indices]
        pids_ord = df_oof['patient_id'].values[ordered_indices]
        labels_ord = df_oof['target_label'].values[ordered_indices]

        sim_matrix = cosine_similarity(Z_ord)
        df_sim = pd.DataFrame(sim_matrix, index=pids_ord, columns=pids_ord)
        sim_tsv_path = os.path.join(insights_dir, f"cosine_similarity_matrix_variant_{variant_name}.tsv")
        df_sim.to_csv(sim_tsv_path, sep='\t')

        # Calcolo coesione intra-classe vs inter-classe
        intra_res, intra_sens, inter_pairs = [], [], []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                s_ij = sim_matrix[i, j]
                if labels_ord[i] == labels_ord[j]:
                    if labels_ord[i] == 1:
                        intra_res.append(s_ij)
                    else:
                        intra_sens.append(s_ij)
                else:
                    inter_pairs.append(s_ij)

        all_intra = intra_res + intra_sens
        if len(all_intra) > 0 and len(inter_pairs) > 0:
            _, p_val = mannwhitneyu(all_intra, inter_pairs, alternative='greater')
            logging.info(
                f"[INSIGHTS Var {variant_name}] Cosine Similarity Cohesion | "
                f"Intra-Resistant: {np.mean(intra_res):.4f} | "
                f"Intra-Sensitive: {np.mean(intra_sens):.4f} | "
                f"Inter-Class: {np.mean(inter_pairs):.4f} | "
                f"Mann-Whitney p={p_val:.4e}"
            )

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='auto')
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

        # Linee di separazione tra classi (es. Sensitive | Resistant)
        for b in class_boundaries[:-1]:
            ax.axhline(b - 0.5, color='black', linewidth=2.0, linestyle='--')
            ax.axvline(b - 0.5, color='black', linewidth=2.0, linestyle='--')

        tick_step = max(1, n_samples // 25)
        tick_pos = list(range(0, n_samples, tick_step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([pids_ord[i] for i in tick_pos], rotation=90, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([pids_ord[i] for i in tick_pos], fontsize=7)

        cls_summary = " | ".join([
            f"{label_encoder_patient.classes_[c]} (n={(labels_ord == c).sum()})"
            for c in sorted(np.unique(labels_ord))
        ])
        ax.set_title(f"Var {variant_name} - Patient Embedding Cosine Similarity\nOrdered by Class [{cls_summary}] & Hierarchical Clustering")
        plt.tight_layout()

        sim_fig_path = os.path.join(insights_dir, f"cosine_similarity_matrix_variant_{variant_name}.png")
        plt.savefig(sim_fig_path, dpi=300)
        plt.close()
        logging.info(f"[INSIGHTS Var {variant_name}] Salvata matrice di similarità coseno in: {sim_fig_path}")

    # ------------------------------------------------------------------
    # 3. IMPORTANZA MEDIA DELLE FEATURE TRA I 5 FOLD
    # ------------------------------------------------------------------
    if feature_importance_records:
        df_imp = pd.DataFrame(feature_importance_records)
        df_imp_agg = (
            df_imp.groupby('feature', as_index=False)
            .agg(
                mean_diff_weight=('diff_weight_res_vs_sens', 'mean'),
                abs_mean_weight=('abs_weight', 'mean'),
                folds_selected=('fold', 'count'),
            )
            .sort_values(by=['folds_selected', 'abs_mean_weight'], ascending=[False, False])
        )
        imp_path = os.path.join(insights_dir, f"top_driving_features_variant_{variant_name}.tsv")
        df_imp_agg.to_csv(imp_path, sep='\t', index=False)
        logging.info(f"[INSIGHTS Var {variant_name}] Salvato ranking feature più informative in: {imp_path}\n")


def compute_effective_mlp_weights(model, feature_names, fold_num):
    """
    Estrae il contributo lineare effettivo dal vettore di input ai logit di classe:
    W_eff = W_in^T @ W_out^T (approssimazione al primo ordine del bottleneck MLP).
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        return []

    w1 = linear_layers[0].weight.detach().cpu().numpy()   # [bottleneck, in_dim]
    w2 = linear_layers[-1].weight.detach().cpu().numpy()  # [num_classes, bottleneck]
    w_eff = w2 @ w1                                       # [num_classes, in_dim]

    if w_eff.shape[0] >= 2:
        diff_w = w_eff[1] - w_eff[0]  # Positivo -> spinge verso classe 1 ('yes' / resistente)
    else:
        diff_w = w_eff[0]

    records = []
    for fname, val in zip(feature_names, diff_w):
        records.append({
            'fold': fold_num,
            'feature': fname,
            'diff_weight_res_vs_sens': float(val),
            'abs_weight': float(abs(val)),
        })
    return records


from sklearn.metrics import roc_curve, f1_score, log_loss
from sklearn.linear_model import LogisticRegression

def find_optimal_threshold(y_true, y_prob_pos, criterion="youden"):
    """
    Stima la soglia decisionale ottimale su predizioni Out-Of-Fold.
    criterion:
      - 'youden': massimizza Sensibilità + Specificità - 1 (TPR - FPR)
      - 'f1_macro': massimizza la Macro F1 su una griglia densa di soglie
    """
    y_true = np.asarray(y_true)
    y_prob_pos = np.asarray(y_prob_pos)

    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
        # Esclude soglie degeneri (> 1.0)
        valid = thresholds <= 1.0
        fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
        best_idx = int(np.argmax(tpr - fpr))
        return float(thresholds[best_idx])

    elif criterion == "f1_macro":
        grid = np.linspace(0.15, 0.85, 141)
        scores = [
            f1_score(y_true, (y_prob_pos >= t).astype(int), average="macro", zero_division=0)
            for t in grid
        ]
        return float(grid[int(np.argmax(scores))])

    raise ValueError(f"Criterio non supportato: {criterion}")


# ==============================================================================
# INIZIALIZZAZIONE PARAMETRI GENERALI E LOGGING
# ==============================================================================

use_metadata = config_file_tree.USE_SAMPLES_METADATA
use_clinical = config_file_tree.USE_CLINICAL_DATA

task = config_file_tree.TASK
sample_target_column = config_file_tree.SAMPLE_TARGET_COL
patient_target_col = config_file_tree.PATIENT_TARGET_COL

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
path_to_clinical = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")

path_train_val_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_train_val.csv")
path_test_csv = os.path.join(config_file_tree.SPLIT_DIR, "samples_test.csv")

path_graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR

gat_pretrained_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)
pretrained_sample_encoder_path = os.path.join(
    gat_pretrained_dir,
    f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt",
)

curr_date = datetime.now().strftime('%d_%m_%Y_hybrid_PatientMLP_NACT_strategyAB')
output_base_dir = os.path.join("trained_models", curr_date)
os.makedirs(output_base_dir, exist_ok=True)

_, config_text = save_training_config_report(output_base_dir)

# GAT Output Paths
gat_output_dir = os.path.join(output_base_dir, "GAT")
gat_learning_curves_dir = os.path.join(gat_output_dir, "learning_curves")
os.makedirs(gat_learning_curves_dir, exist_ok=True)

gat_global_model_path = os.path.join(gat_output_dir, f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt")
label_encoder_sample_path = os.path.join(gat_output_dir, f"label_encoder_{sample_target_column}.joblib")
metadata_preprocessors_path = os.path.join(gat_output_dir, f"metadata_preprocessors_{sample_target_column}.joblib")
node_scaler_path = os.path.join(gat_output_dir, f"node_scaler_{sample_target_column}.joblib")
gat_report_filename = "report_kfold_test_results_GAT.csv"

# MLP Output Paths (separati per Variante A e Variante B)
mlp_base_output_dir = os.path.join(output_base_dir, "MLP")
label_encoder_patient_path = os.path.join(mlp_base_output_dir, f"label_encoder_patient_{patient_target_col}.joblib")

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    os.makedirs(mlp_base_output_dir, exist_ok=True)
    if not config_file_tree.SAMPLES_LEVEL_TRAIN:
        handle_pretrained_gat_checkpoint(
            gat_pretrained_dir=gat_pretrained_dir,
            gat_output_dir=gat_output_dir,
            config_text=config_text,
        )

active_feature_names = [config_file_tree.NODE_FEATURE_NAMES[i] for i in config_file_tree.NODE_FEATURE_INDEX]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_base_dir, 'last_execution.log'), mode='w'),
        logging.StreamHandler(),
    ],
    force=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    cpus_allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    try:
        torch.set_num_threads(cpus_allocated)
        torch.set_num_interop_threads(max(1, cpus_allocated // 2))
    except RuntimeError as e:
        logging.warning(f"Impossibile impostare interop_threads: {e}")
    logging.info(f"Allocated threads for PyTorch: {torch.get_num_threads()}")

logging.info(f"Device: {device} | Active Patient Aggregation Variants: {VARIANTS_TO_RUN}\n")


# ==============================================================================
# CARICAMENTO METADATI, DATI CLINICI E GRAFI
# ==============================================================================

df_metadata = pd.read_csv(path_to_metadata, sep="\t")
metadata_train_val_samples = pd.read_csv(path_train_val_csv)
metadata_test_samples = pd.read_csv(path_test_csv)

df_metadata_train_val = df_metadata[
    df_metadata['sample_id'].isin(metadata_train_val_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

df_metadata_test = df_metadata[
    df_metadata['sample_id'].isin(metadata_test_samples['sample_id'])
].dropna(subset=[sample_target_column]).copy()

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    cols_to_keep = ['patient_id', patient_target_col]
    if use_clinical:
        cols_to_keep += config_file_tree.CLINICAL_FEATURES
    df_clinical = pd.read_csv(path_to_clinical, sep="\t", usecols=cols_to_keep)
    clinical_patient_id_col = 'patient_id'

if config_file_tree.SAMPLES_LEVEL_TRAIN:
    label_encoder_sample = sklearn.preprocessing.LabelEncoder()
    label_encoder_sample.fit(df_metadata[sample_target_column].dropna())
    df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(df_metadata_train_val[sample_target_column])
    df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])
    joblib.dump(label_encoder_sample, label_encoder_sample_path)
else:
    if os.path.exists(label_encoder_sample_path):
        label_encoder_sample = joblib.load(label_encoder_sample_path)
        df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(df_metadata_train_val[sample_target_column])
        df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])
    else:
        label_encoder_sample = LabelEncoder()
        label_encoder_sample.fit(df_metadata[sample_target_column].dropna())
        df_metadata_train_val[sample_target_column] = label_encoder_sample.transform(df_metadata_train_val[sample_target_column])
        df_metadata_test[sample_target_column] = label_encoder_sample.transform(df_metadata_test[sample_target_column])
        joblib.dump(label_encoder_sample, label_encoder_sample_path)

sample_class_names = [str(c) for c in label_encoder_sample.classes_]
logging.info(f"Sample classes ({sample_target_column}): {sample_class_names}")

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    df_clinical_clean = df_clinical.dropna(subset=[patient_target_col]).copy()
    label_encoder_patient = sklearn.preprocessing.LabelEncoder()
    df_clinical_clean[patient_target_col] = label_encoder_patient.fit_transform(df_clinical_clean[patient_target_col])

    num_classes_patient_target = len(label_encoder_patient.classes_)
    joblib.dump(label_encoder_patient, label_encoder_patient_path)
    logging.info(f"Patient classes ({patient_target_col}): {label_encoder_patient.classes_}\n")

    if config_file_tree.RUN_PERMUTATION_TEST:
        seed_perm = 42
        logging.info("=======================================================")
        logging.info(f"[PERMUTATION TEST] ACTIVE (seed={seed_perm})")
        np.random.seed(seed_perm)
        original_labels = df_clinical_clean[patient_target_col].values.copy()
        df_clinical_clean[patient_target_col] = np.random.permutation(original_labels)
        logging.info(f"Original distribution: {np.bincount(original_labels)}")
        logging.info(f"Permuted distribution: {np.bincount(df_clinical_clean[patient_target_col].values)}")
        logging.info("=======================================================\n")

    patient_target_map = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()

logging.info("Train + Val Dataset init...")
raw_train_val_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_train_val, task=task, target_col=sample_target_column)
logging.info("Test Dataset init...")
raw_test_dataset = PatientSampleDataset(path_graphs_dir, df_metadata_test, task=task, target_col=sample_target_column)

logging.info("Pre-loading datasets into RAM...")
train_val_dataset = [raw_train_val_dataset[i] for i in tqdm(range(len(raw_train_val_dataset)), desc="Loading Train/Val")]
test_dataset = [raw_test_dataset[i] for i in tqdm(range(len(raw_test_dataset)), desc="Loading Test")]

print_details(raw_train_val_dataset)
num_genes_features = raw_train_val_dataset.num_features

for graph in train_val_dataset:
    graph.x_raw = graph.x.clone()
for graph in test_dataset:
    graph.x_raw = graph.x.clone()

epochs = 50
y_labels_samples = torch.tensor(df_metadata_train_val[sample_target_column].values, dtype=torch.long)
num_classes_samples = len(torch.unique(y_labels_samples))
class_counts_samples = torch.bincount(y_labels_samples)
class_weights_samples = (len(y_labels_samples) / (len(class_counts_samples) * class_counts_samples.float())).to(device)
criterion_samples = torch.nn.CrossEntropyLoss(weight=class_weights_samples)


# ==============================================================================
# STRATIFIED K-FOLD LOOP
# ==============================================================================

groups = df_metadata_train_val['patient_id'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    patient_to_clinical_target = df_clinical_clean.set_index(clinical_patient_id_col)[patient_target_col].to_dict()
    y_skf = [
        f"{row[sample_target_column]}_{patient_to_clinical_target.get(row['patient_id'], 'unfiltered')}"
        for _, row in df_metadata_train_val.iterrows()
    ]
else:
    y_skf = y_labels_samples

samples_fold_results = []
global_best_gat_score = float('inf')

# Strutture di raccolta risultati separate per ciascuna variante ("A", "B")
variant_trackers = {
    var: {
        'fold_results': [],
        'subgroup_results': [],
        'oof_embeddings': [],
        'feature_importances': [],
        'global_best_loss': float('inf'),
        'out_dir': os.path.join(mlp_base_output_dir, f"Variant_{var}"),
    }
    for var in VARIANTS_TO_RUN
}
for var in VARIANTS_TO_RUN:
    os.makedirs(os.path.join(variant_trackers[var]['out_dir'], "learning_curves"), exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_val_dataset, y_skf, groups=groups)):
    logging.info(f"==================== FOLD {fold + 1} ====================\n")

    # 1. SCALING NODE FEATURES (fit solo su train_idx)
    train_nodes_np = np.vstack([train_val_dataset[i].x_raw.numpy() for i in train_idx])
    node_scaler = SelectiveNodeScaler(active_feature_names=active_feature_names)
    node_scaler.fit(train_nodes_np)

    for i in train_idx:
        train_val_dataset[i].x = torch.tensor(node_scaler.transform(train_val_dataset[i].x_raw.numpy()), dtype=torch.float32)
    for i in val_idx:
        train_val_dataset[i].x = torch.tensor(node_scaler.transform(train_val_dataset[i].x_raw.numpy()), dtype=torch.float32)
    for graph in test_dataset:
        graph.x = torch.tensor(node_scaler.transform(graph.x_raw.numpy()), dtype=torch.float32)

    # 2. METADATA SAMPLE-LEVEL PROCESSING
    num_sample_features = 0
    fold_metadata_preprocessors = None
    if use_metadata:
        df_metadata_train_raw = df_metadata_train_val.iloc[train_idx].copy()
        df_metadata_val_raw = df_metadata_train_val.iloc[val_idx].copy()
        df_metadata_test_raw = df_metadata_test.copy()

        df_train_proc, encoders, scalers, imputers, categories = pre_train_imputation_encoding(
            df=df_metadata_train_raw, task=task, target_col=sample_target_column, is_train=True
        )
        fold_metadata_preprocessors = {'encoders': encoders, 'scalers': scalers, 'imputers': imputers, 'categories': categories}
        df_val_proc = pre_train_imputation_encoding(
            df=df_metadata_val_raw, task=task, target_col=sample_target_column, is_train=False,
            encoders=encoders, scalers=scalers, imputers=imputers, categories=categories
        )
        df_test_proc = pre_train_imputation_encoding(
            df=df_metadata_test_raw, task=task, target_col=sample_target_column, is_train=False,
            encoders=encoders, scalers=scalers, imputers=imputers, categories=categories
        )

        df_train_val_proc = pd.DataFrame(pd.concat([df_train_proc, df_val_proc], axis=0)).set_index('sample_id')
        df_test_proc = df_test_proc.set_index('sample_id')
        exclude_cols = {sample_target_column, 'patient_id'}
        sample_feature_cols = [c for c in df_train_val_proc.columns if c not in exclude_cols]
        num_sample_features = len(sample_feature_cols)

        update_graphs_metadata(train_val_dataset, df_train_val_proc, sample_feature_cols)
        update_graphs_metadata(test_dataset, df_test_proc, sample_feature_cols)

    # 3. SAMPLE ENCODER (GAT)
    num_nodes_per_graph = raw_train_val_dataset[0].x.shape[0]
    model_samples_encoder = GAT(
        num_node_features=num_genes_features,
        num_classes=num_classes_samples,
        num_nodes=num_nodes_per_graph,
        num_sample_features=num_sample_features if use_metadata else 0,
    ).to(device)

    train_loader = DataLoader(Subset(train_val_dataset, train_idx), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(train_val_dataset, val_idx), batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    if config_file_tree.SAMPLES_LEVEL_TRAIN:
        optimizer_samples = torch.optim.AdamW(model_samples_encoder.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler_samples = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_samples, mode='min', factor=0.5, patience=10)
        best_val_loss = float('inf')
        early_stopping_counter = 0
        history_gat = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        fold_best_gat_weights = None

        for epoch in tqdm(range(1, epochs + 1), desc=f"GAT Fold {fold + 1}"):
            train_loss, train_acc = train_epoch(device, model_samples_encoder, train_loader, optimizer_samples, criterion_samples)
            val_loss, val_metrics = evaluate(device, model_samples_encoder, val_loader, criterion_samples)
            scheduler_samples.step(val_loss)

            history_gat['train_loss'].append(train_loss)
            history_gat['val_loss'].append(val_loss)
            history_gat['train_metric'].append(train_acc)
            history_gat['val_metric'].append(val_metrics['acc'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                fold_best_gat_weights = {k: v.cpu().clone() for k, v in model_samples_encoder.state_dict().items()}
                early_stopping_counter = 0
                if best_val_loss < global_best_gat_score:
                    global_best_gat_score = best_val_loss
                    torch.save(fold_best_gat_weights, gat_global_model_path)
                    joblib.dump(node_scaler, node_scaler_path)
                    if use_metadata and fold_metadata_preprocessors is not None:
                        joblib.dump(fold_metadata_preprocessors, metadata_preprocessors_path)
            else:
                early_stopping_counter += 1
            if early_stopping_counter > 15:
                break

        save_learning_curves(history_gat, fold_num=fold + 1, output_dir=gat_learning_curves_dir)
        if fold_best_gat_weights is not None:
            model_samples_encoder.load_state_dict(fold_best_gat_weights)

        test_loss, test_metrics = evaluate(device, model_samples_encoder, test_loader, criterion_samples)
        samples_fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
        })
    else:
        if os.path.exists(pretrained_sample_encoder_path):
            model_samples_encoder.load_state_dict(torch.load(pretrained_sample_encoder_path, map_location=device))
        else:
            raise FileNotFoundError(f"Pre-trained model not found at {pretrained_sample_encoder_path}")

    # 4. PATIENT LEVEL TRAINING CON PatientMLP (VARIANTI A E B)
    if config_file_tree.PATIENTS_LEVEL_TRAIN:
        logging.info(f"--- PATIENT-LEVEL MLP TRAINING (Fold {fold + 1}) ---\n")

        patient_clinical_map = {}
        patients_feature_cols = []
        if use_clinical:
            train_patients_ids = set(df_metadata_train_val.iloc[train_idx]['patient_id'].unique())
            val_patients_ids = set(df_metadata_train_val.iloc[val_idx]['patient_id'].unique())
            test_patients_ids = set(df_metadata_test['patient_id'].unique())

            df_clin_train = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(train_patients_ids)].copy()
            df_clin_val = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(val_patients_ids)].copy()
            df_clin_test = df_clinical_clean[df_clinical_clean[clinical_patient_id_col].isin(test_patients_ids)].copy()

            df_clin_train_p, p_enc, p_sca, p_imp, p_cat = pre_train_imputation_encoding(
                df=df_clin_train, task='classification', target_col=patient_target_col, is_train=True
            )
            df_clin_val_p = pre_train_imputation_encoding(
                df=df_clin_val, task='classification', target_col=patient_target_col, is_train=False,
                encoders=p_enc, scalers=p_sca, imputers=p_imp, categories=p_cat
            )
            df_clin_test_p = pre_train_imputation_encoding(
                df=df_clin_test, task='classification', target_col=patient_target_col, is_train=False,
                encoders=p_enc, scalers=p_sca, imputers=p_imp, categories=p_cat
            )

            df_clin_all = pd.concat([df_clin_train_p, df_clin_val_p, df_clin_test_p], axis=0).set_index(clinical_patient_id_col)
            exclude_p_cols = {patient_target_col, 'patient_id'}
            patients_feature_cols = [c for c in df_clin_all.columns if c not in exclude_p_cols]

            for pid in df_clin_all.index:
                feats = df_clin_all.loc[pid, patients_feature_cols].values.astype(np.float32)
                patient_clinical_map[str(pid).strip()] = torch.tensor(feats, dtype=torch.float32)

        train_samples = [train_val_dataset[i] for i in train_idx]
        val_samples = [train_val_dataset[i] for i in val_idx]

        # Se la Variante A è richiesta, estraiamo gli embedding dalla GAT congelata una sola volta
        sample_emb_map = None
        if "A" in VARIANTS_TO_RUN:
            all_fold_samples = train_samples + val_samples + list(test_dataset)
            sample_emb_map = extract_sample_gat_embeddings(device, model_samples_encoder, all_fold_samples)

        for variant in VARIANTS_TO_RUN:
            logging.info(f">>> [Fold {fold + 1} | Variant {variant}] Costruzione vettori aggregati a livello paziente...")

            if variant == "A":
                X_tr, y_tr, pids_tr, feat_names, _, tissue_means = build_patient_vectors_variant_a(
                    train_samples, patient_target_map, sample_emb_map,
                    num_tissue_classes=num_classes_samples,
                    tissue_class_names=sample_class_names,
                    patient_clinical_map=patient_clinical_map if use_clinical else None,
                    clinical_feature_names=patients_feature_cols,
                    fit_tissue_means=DECONFOUND_TISSUE_IN_VAR_A,
                )
                X_va, y_va, pids_va, _, dom_tissues_va, _ = build_patient_vectors_variant_a(
                    val_samples, patient_target_map, sample_emb_map,
                    num_tissue_classes=num_classes_samples,
                    tissue_class_names=sample_class_names,
                    patient_clinical_map=patient_clinical_map if use_clinical else None,
                    clinical_feature_names=patients_feature_cols,
                    tissue_means_train=tissue_means if DECONFOUND_TISSUE_IN_VAR_A else None,
                    fit_tissue_means=False,
                )
                X_te, y_te, pids_te, _, dom_tissues_te, _ = build_patient_vectors_variant_a(
                    test_dataset, patient_target_map, sample_emb_map,
                    num_tissue_classes=num_classes_samples,
                    tissue_class_names=sample_class_names,
                    patient_clinical_map=patient_clinical_map if use_clinical else None,
                    clinical_feature_names=patients_feature_cols,
                    tissue_means_train=tissue_means if DECONFOUND_TISSUE_IN_VAR_A else None,
                    fit_tissue_means=False,
                )

            elif variant == "B":
                X_tr, y_tr, pids_tr, feat_names, _, omic_selector = build_patient_vectors_variant_b(
                    train_samples, patient_target_map, active_feature_names,
                    num_tissue_classes=num_classes_samples,
                    tissue_class_names=sample_class_names,
                    patient_clinical_map=patient_clinical_map if use_clinical else None,
                    clinical_feature_names=patients_feature_cols,
                    top_k=TOP_K_OMIC_FEATURES,
                    fit_selector=True,
                )
                X_va, y_va, pids_va, _, dom_tissues_va, _ = build_patient_vectors_variant_b(
                    val_samples, patient_target_map, active_feature_names,
                    num_tissue_classes=num_classes_samples,
                    tissue_class_names=sample_class_names,
                    patient_clinical_map=patient_clinical_map if use_clinical else None,
                    clinical_feature_names=patients_feature_cols,
                    selector=omic_selector,
                    fit_selector=False,
                )
                X_te, y_te, pids_te, _, dom_tissues_te, _ = build_patient_vectors_variant_b(
                    test_dataset, patient_target_map, active_feature_names,
                    num_tissue_classes=num_classes_samples,
                    tissue_class_names=sample_class_names,
                    patient_clinical_map=patient_clinical_map if use_clinical else None,
                    clinical_feature_names=patients_feature_cols,
                    selector=omic_selector,
                    fit_selector=False,
                )
            else:
                raise ValueError(f"Variante non supportata: {variant}. Usa 'A' o 'B'.")

            # Standardizzazione fittata SOLO su X_tr del fold corrente
            feat_scaler = StandardScaler()
            X_tr_scaled = feat_scaler.fit_transform(X_tr)
            X_va_scaled = feat_scaler.transform(X_va)
            X_te_scaled = feat_scaler.transform(X_te)

            # Test confronto con Linear Probe L2:
            # Da inserire dentro `for variant in VARIANTS_TO_RUN:` subito dopo:
            #   X_tr_scaled = feat_scaler.fit_transform(X_tr)
            #   X_va_scaled = feat_scaler.transform(X_va)
            #   X_te_scaled = feat_scaler.transform(X_te)

            if variant == "A":
                # 1) Tuning leggero di C su Validation (log_loss) oppure C fisso = 0.1
                candidate_C = [0.01, 0.05, 0.1, 0.5, 1.0]
                best_c, best_va_loss, best_lr_model = 0.1, float("inf"), None

                for c_val in candidate_C:
                    lr_tmp = LogisticRegression(
                        penalty="l2",
                        C=c_val,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42,
                    )
                    lr_tmp.fit(X_tr_scaled, y_tr)
                    probs_va_lr = lr_tmp.predict_proba(X_va_scaled)
                    va_loss_lr = log_loss(y_va, probs_va_lr, labels=[0, 1])

                    if va_loss_lr < best_va_loss:
                        best_va_loss = va_loss_lr
                        best_c = c_val
                        best_lr_model = lr_tmp

                # 2) Valutazione su Test set con le stesse metriche dell'MLP
                probs_te_lr = best_lr_model.predict_proba(X_te_scaled)
                preds_te_lr = best_lr_model.predict(X_te_scaled)
                test_loss_lr = log_loss(y_te, probs_te_lr, labels=[0, 1])
                lr_metrics = compute_classification_metrics(y_te, preds_te_lr, probs_te_lr)

                # Salva OOF probabilities del Linear Probe (utile anche per soglia OOF!)
                probs_va_best_lr = best_lr_model.predict_proba(X_va_scaled)
                for idx_p, pid in enumerate(pids_va):
                    variant_trackers[variant].setdefault('lr_oof_records', []).append({
                        'patient_id': str(pid).strip(),
                        'fold': fold + 1,
                        'target_label': int(y_va[idx_p]),
                        'prob_class_1': float(probs_va_best_lr[idx_p, 1]),
                    })

                variant_trackers[variant].setdefault('lr_fold_results', []).append({
                    'fold': fold + 1,
                    'best_C': best_c,
                    'test_loss': test_loss_lr,
                    'acc': lr_metrics['acc'],
                    'f1': lr_metrics['f1'],
                    'auc': lr_metrics['auc'],
                })

                # Estrazione esatta dei pesi lineari (senza approssimazione al primo ordine)
                for fname, w_val in zip(feat_names, best_lr_model.coef_[0]):
                    variant_trackers[variant].setdefault('lr_feature_importances', []).append({
                        'fold': fold + 1,
                        'feature': fname,
                        'weight_res_vs_sens': float(w_val),
                        'abs_weight': float(abs(w_val)),
                    })

                logging.info(
                    f"[LinearProbe Var A | C={best_c}] Fold {fold + 1} Test Loss: {test_loss_lr:.4f} | "
                    f"Acc: {lr_metrics['acc']:.4f} | F1: {lr_metrics['f1']:.4f} | AUC: {lr_metrics['auc']:.4f}"
                )

            X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32, device=device)
            y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
            X_va_t = torch.tensor(X_va_scaled, dtype=torch.float32, device=device)
            y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
            X_te_t = torch.tensor(X_te_scaled, dtype=torch.float32, device=device)
            y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

            # Pesi di classe calcolati sui pazienti di Train del fold
            class_counts_p = torch.clamp(torch.bincount(y_tr_t, minlength=num_classes_patient_target), min=1)
            class_weights_p = (len(y_tr_t) / (len(class_counts_p) * class_counts_p.float())).to(device)
            criterion_p = nn.CrossEntropyLoss(weight=class_weights_p)

            in_dim = X_tr_t.shape[1]
            mlp_model = PatientMLP(
                in_dim=in_dim,
                bottleneck_dim=MLP_BOTTLENECK_DIM,
                num_classes=num_classes_patient_target,
                dropout=MLP_DROPOUT,
            ).to(device)

            if fold == 0:
                n_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
                logging.info(f"[Variant {variant}] Input dim: {in_dim} | Trainable MLP params: {n_params}")

            optimizer_mlp = torch.optim.AdamW(mlp_model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
            scheduler_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mlp, T_max=MLP_EPOCHS, eta_min=1e-5)

            best_ema_val_loss = float('inf')
            ema_val_loss = None
            early_stop_cnt = 0
            fold_best_weights = None
            history_mlp = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

            for epoch in range(1, MLP_EPOCHS + 1):
                # Full-batch training step (ideale per N_train ~ 54 pazienti)
                mlp_model.train()
                optimizer_mlp.zero_grad()
                logits_tr, _ = _forward_mlp(mlp_model, X_tr_t)
                loss_tr = criterion_p(logits_tr, y_tr_t)
                loss_tr.backward()
                torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
                optimizer_mlp.step()
                scheduler_mlp.step()

                train_acc = (logits_tr.argmax(dim=1) == y_tr_t).float().mean().item()

                # Validation step
                mlp_model.eval()
                with torch.no_grad():
                    logits_va, _ = _forward_mlp(mlp_model, X_va_t)
                    loss_va = criterion_p(logits_va, y_va_t).item()
                    probs_va = torch.softmax(logits_va, dim=1).cpu().numpy()
                    preds_va = logits_va.argmax(dim=1).cpu().numpy()
                    va_metrics = compute_classification_metrics(y_va, preds_va, probs_va)

                ema_val_loss = loss_va if ema_val_loss is None else (EMA_ALPHA * loss_va + (1.0 - EMA_ALPHA) * ema_val_loss)

                history_mlp['train_loss'].append(loss_tr.item())
                history_mlp['val_loss'].append(loss_va)
                history_mlp['train_metric'].append(train_acc)
                history_mlp['val_metric'].append(va_metrics['acc'])

                logging.info(
                    f"[MLP Var {variant}] Epoch {epoch:02d} | Train Loss: {loss_tr.item():.4f}, Train Acc: {train_acc:.4f} | Val Loss: {loss_va:.4f}")
                logging.info(
                    f"Val metrics: Acc: {va_metrics['acc']:.4f} | F1: {va_metrics['f1']:.4f}, AUC: {va_metrics['auc']:.4f}"
                    )

                if ema_val_loss < best_ema_val_loss:
                    logging.info("PATIENT LEVEL IMPROVEMENT!\n")
                    best_ema_val_loss = ema_val_loss
                    fold_best_weights = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}
                    early_stop_cnt = 0

                    if best_ema_val_loss < variant_trackers[variant]['global_best_loss']:
                        variant_trackers[variant]['global_best_loss'] = best_ema_val_loss
                        best_ckpt_path = os.path.join(
                            variant_trackers[variant]['out_dir'],
                            f"best_MLP_var{variant}_{patient_target_col}.pt",
                        )
                        torch.save(fold_best_weights, best_ckpt_path)
                else:
                    early_stop_cnt += 1

                if early_stop_cnt >= MLP_EARLY_STOPPING_PATIENCE:
                    break

            var_out_dir = variant_trackers[variant]['out_dir']
            save_learning_curves(history_mlp, fold_num=fold + 1, output_dir=os.path.join(var_out_dir, "learning_curves"))

            if fold_best_weights is not None:
                mlp_model.load_state_dict(fold_best_weights)

            # 1) Estrazione Out-Of-Fold Embeddings su Validation (zero data leakage)
            mlp_model.eval()
            with torch.no_grad():
                logits_va_best, z_va_best = _forward_mlp(mlp_model, X_va_t)
                probs_va_best = torch.softmax(logits_va_best, dim=1).cpu().numpy()
                z_va_np = z_va_best.cpu().numpy()

            for idx_p, pid in enumerate(pids_va):
                yt = int(y_va[idx_p])
                rec = {
                    'patient_id': str(pid).strip(),
                    'fold': fold + 1,
                    'target_label': yt,
                    'target_label_name': str(label_encoder_patient.classes_[yt]),
                    'prob_class_1': float(probs_va_best[idx_p, 1]) if probs_va_best.shape[1] > 1 else float(1.0 - probs_va_best[idx_p, 0]),
                    'dominant_tissue': dom_tissues_va[idx_p],
                }
                for dim_i in range(z_va_np.shape[1]):
                    rec[f"emb_{dim_i}"] = float(z_va_np[idx_p, dim_i])
                variant_trackers[variant]['oof_embeddings'].append(rec)

            # 2) Estrazione importanza feature effettive del fold
            variant_trackers[variant]['feature_importances'].extend(
                compute_effective_mlp_weights(mlp_model, feat_names, fold_num=fold + 1)
            )

            # 3) Valutazione su Test set
            with torch.no_grad():
                logits_te, _ = _forward_mlp(mlp_model, X_te_t)
                test_loss = criterion_p(logits_te, y_te_t).item()
                probs_te = torch.softmax(logits_te, dim=1).cpu().numpy()
                preds_te = logits_te.argmax(dim=1).cpu().numpy()
                test_metrics = compute_classification_metrics(y_te, preds_te, probs_te)
                variant_trackers[variant].setdefault('test_probs_per_fold', []).append({
                    'fold': fold + 1,
                    'y_te': y_te,
                    'probs_te': probs_te,
                })

            patient_counts = np.bincount(y_te, minlength=num_classes_patient_target)
            fold_patient_dict = {
                'fold': fold + 1,
                'total_patients': len(y_te),
            }
            for cls_idx, cls_name in enumerate(label_encoder_patient.classes_):
                fold_patient_dict[f'n_{cls_name}'] = int(patient_counts[cls_idx])

            fold_patient_dict.update({
                'test_loss': test_loss,
                'acc': test_metrics['acc'],
                'f1': test_metrics['f1'],
                'auc': test_metrics['auc'],
            })
            variant_trackers[variant]['fold_results'].append(fold_patient_dict)

            logging.info(
                f"[MLP Var {variant}] Fold {fold + 1} Test Loss: {test_loss:.4f} | "
                f"Acc: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}"
            )

            _, subgroup_metrics = save_mlp_patient_predictions(
                patient_ids=pids_te,
                y_true=y_te,
                probs=probs_te,
                dominant_tissues=dom_tissues_te,
                label_encoder_patient=label_encoder_patient,
                fold_num=fold + 1,
                output_dir=var_out_dir,
                clinical_features_to_include=['Treatment_strategy', 'PFI_KaplanM_allHGSC'],
            )
            for grp_name, g_m in subgroup_metrics.items():
                variant_trackers[variant]['subgroup_results'].append({
                    'fold': fold + 1,
                    'subgroup': grp_name,
                    'n_patients': g_m['n_patients'],
                    'acc': g_m['acc'],
                    'f1': g_m['f1'],
                    'auc': g_m['auc'],
                })


# ==============================================================================
# RIEPILOGO FINALE & GENERAZIONE INSIGHTS SUGLI EMBEDDING
# ==============================================================================

if config_file_tree.SAMPLES_LEVEL_TRAIN and samples_fold_results:
    logging.info("--- SAMPLE LEVEL RESULTS (GAT) ---")
    show_fold_results(pd.DataFrame(samples_fold_results), task, save_to_csv=True, dir_to_save=gat_output_dir, filename=gat_report_filename)

if config_file_tree.PATIENTS_LEVEL_TRAIN:
    for variant in VARIANTS_TO_RUN:
        tracker = variant_trackers[variant]
        var_out_dir = tracker['out_dir']

        if tracker['fold_results']:
            logging.info(f"================ PATIENT LEVEL RESULTS: VARIANT {variant} ================")
            show_fold_results(
                pd.DataFrame(tracker['fold_results']),
                task,
                save_to_csv=True,
                dir_to_save=var_out_dir,
                filename=f"report_kfold_test_results_MLP_var{variant}.csv",
            )

        if tracker['subgroup_results']:
            df_sub = pd.DataFrame(tracker['subgroup_results'])
            df_sub.to_csv(os.path.join(var_out_dir, f"report_kfold_subgroups_MLP_var{variant}.csv"), index=False)
            for grp, df_grp in df_sub.groupby('subgroup'):
                logging.info(
                    f"[Var {variant}] Subgroup [{grp}] | "
                    f"AUC: {df_grp['auc'].mean():.4f} ± {df_grp['auc'].var():.4f} | "
                    f"F1: {df_grp['f1'].mean():.4f} ± {df_grp['f1'].var():.4f}"
                )

        if tracker['oof_embeddings']:
            extract_and_save_embedding_insights(
                oof_records=tracker['oof_embeddings'],
                feature_importance_records=tracker['feature_importances'],
                label_encoder_patient=label_encoder_patient,
                variant_name=variant,
                output_dir=var_out_dir,
            )

        if tracker['oof_embeddings'] and 'test_probs_per_fold' in tracker:
            df_oof_all = pd.DataFrame(tracker['oof_embeddings'])
            calibrated_rows = []

            for item in tracker['test_probs_per_fold']:
                f_num = item['fold']
                y_te_f = item['y_te']
                probs_te_f = item['probs_te']

                # Stima soglia SOLO sui 4 fold diversi da f_num (zero leakage)
                df_oof_other = df_oof_all[df_oof_all['fold'] != f_num]
                tau_k = find_optimal_threshold(
                    df_oof_other['target_label'].values,
                    df_oof_other['prob_class_1'].values,
                    criterion="youden",  # oppure "f1_macro"
                )

                preds_cal = (probs_te_f[:, 1] >= tau_k).astype(int)
                m_cal = compute_classification_metrics(y_te_f, preds_cal, probs_te_f)
                calibrated_rows.append({
                    'fold': f_num,
                    'threshold': tau_k,
                    'acc_cal': m_cal['acc'],
                    'f1_cal': m_cal['f1'],
                    'auc': m_cal['auc'],
                })

            df_cal = pd.DataFrame(calibrated_rows)
            logging.info(
                f"[Var {variant} | OOF-Calibrated] "
                f"Mean Thresh: {df_cal['threshold'].mean():.4f} | "
                f"Acc: {df_cal['acc_cal'].mean():.4f} ± {df_cal['acc_cal'].var():.4f} | "
                f"F1: {df_cal['f1_cal'].mean():.4f} ± {df_cal['f1_cal'].var():.4f}"
            )

            # Ensemble dei 5 fold sul Test set con soglia globale OOF
            tau_global = find_optimal_threshold(
                df_oof_all['target_label'].values,
                df_oof_all['prob_class_1'].values,
                criterion="youden",
            )
            mean_probs_te = np.mean([x['probs_te'] for x in tracker['test_probs_per_fold']], axis=0)
            y_te_ref = tracker['test_probs_per_fold'][0]['y_te']
            ens_preds = (mean_probs_te[:, 1] >= tau_global).astype(int)
            ens_metrics = compute_classification_metrics(y_te_ref, ens_preds, mean_probs_te)
            logging.info(
                f"[Var {variant} | 5-Fold Ensemble @ tau={tau_global:.4f}] "
                f"Test AUC: {ens_metrics['auc']:.4f} | F1: {ens_metrics['f1']:.4f} | Acc: {ens_metrics['acc']:.4f}"
            )

        if 'lr_fold_results' in tracker and tracker['lr_fold_results']:
            logging.info(f"================ LINEAR PROBE L2 RESULTS: VARIANT {variant} ================")
            show_fold_results(
                pd.DataFrame(tracker['lr_fold_results']),
                task,
                save_to_csv=True,
                dir_to_save=var_out_dir,
                filename=f"report_kfold_test_results_LinearProbe_var{variant}.csv",
            )

logging.info("DONE\n")
