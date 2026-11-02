import os

import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from torch_geometric.loader import DataLoader
import umap

import config_file_tree
from models.GAT import GAT
from PatientSampleDataset import PatientSampleDataset, update_graphs_metadata

# ---------------------------------------------------------
# 1. Configurazione Path e Modello GAT
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sample_target_column = config_file_tree.SAMPLE_TARGET_COL
use_metadata = config_file_tree.USE_SAMPLES_METADATA
gat_pretrained_dir = os.path.join(config_file_tree.TRAINED_MODELS_DIR, config_file_tree.SAMPLE_ENCODER_FOLDER)

pretrained_sample_encoder_path = os.path.join(
    gat_pretrained_dir,
    f"best_GAT_{sample_target_column}_{'withMeta' if use_metadata else 'noMeta'}.pt"
)

path_to_metadata = os.path.join(config_file_tree.SPLIT_DIR, "metadata_for_training.tsv")
path_to_clinical = os.path.join(config_file_tree.SPLIT_DIR, "clinical_data_for_training.tsv")

# Caricamento Metadati e Dati Clinici dello Split
df_metadata = pd.read_csv(path_to_metadata, sep='\t')
df_clinical = pd.read_csv(path_to_clinical, sep='\t')

# Merge tra metadati dei campioni e dati clinici dei pazienti filtrati
df_merged = df_metadata.merge(df_clinical, on='patient_id', how='left')
# Conserviamo le etichette testuali trasparenti per la visualizzazione
df_merged['tissue_type_str'] = df_merged[sample_target_column].astype(str)

# ---------------------------------------------------------
# Codifica del Target e Caricamento Preprocessori
# ---------------------------------------------------------
label_encoder_path = os.path.join(gat_pretrained_dir, f"label_encoder_{sample_target_column}.joblib")
if os.path.exists(label_encoder_path):
    label_encoder_sample = joblib.load(label_encoder_path)
    df_merged[sample_target_column] = label_encoder_sample.transform(df_merged[sample_target_column])
    sample_class_names = [str(c) for c in label_encoder_sample.classes_]
else:
    # Fallback se non presente il file joblib
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_merged[sample_target_column] = le.fit_transform(df_merged[sample_target_column])
    sample_class_names = [str(c) for c in le.classes_]

# ---------------------------------------------------------
# 2. Caricamento Dataset PyG ed Estrazione Embedding dal GAT
# ---------------------------------------------------------
print("Loading graphs dataset and extracting embeddings from GAT...")
graphs_dir = config_file_tree.PROCESSED_GRAPHS_DIR
raw_dataset = PatientSampleDataset(
    graphs_dir=graphs_dir,
    samples_metadata_df=df_merged,
    task=config_file_tree.TASK,
    target_col=sample_target_column
)

# Applica pre-processing / scalatura feature nodi se presente
dataset = [raw_dataset[i] for i in range(len(raw_dataset))]

# Applicazione dello Scaler dei nodi (se salvato)
node_scaler_path = os.path.join(gat_pretrained_dir, f"node_scaler_{sample_target_column}.joblib")
if os.path.exists(node_scaler_path):
    print("Applying saved node scaler...")
    node_scaler = joblib.load(node_scaler_path)
    for graph in dataset:
        graph.x = torch.tensor(node_scaler.transform(graph.x.numpy()), dtype=torch.float32)

# Applicazione metadata preprocessors se richiesto
num_sample_features = 0
if use_metadata:
    meta_prep_path = os.path.join(gat_pretrained_dir, f"metadata_preprocessors_{sample_target_column}.joblib")
    if os.path.exists(meta_prep_path):
        prep_dict = joblib.load(meta_prep_path)
        update_graphs_metadata(
            dataset,
            df_merged,
            config_file_tree.METADATA_FEATURES,
            prep_dict.get('num_scaler'),
            prep_dict.get('cat_encoder'),
            prep_dict.get('imputer')
        )
        if len(dataset) > 0 and hasattr(dataset[0], 'metadata_features'):
            num_sample_features = dataset[0].metadata_features.size(1)


sample_0 = dataset[0]
num_node_features = sample_0.x.size(1)
num_unique_genes = sample_0.x.size(0)

# Inizializzazione e caricamento pesi GAT
model = GAT(
    num_node_features=num_node_features,
    num_classes=len(sample_class_names),
    num_nodes=num_unique_genes,
    num_sample_features=num_sample_features
).to(device)

if os.path.exists(pretrained_sample_encoder_path):
    model.load_state_dict(torch.load(pretrained_sample_encoder_path, map_location=device))
    print(f"Successfully loaded pretrained GAT weights from: {pretrained_sample_encoder_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found at: {pretrained_sample_encoder_path}")

model.eval()

# Forward Pass per estrarre gli embedding
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
sample_embeddings_list = []

with torch.no_grad():
    for batch in data_loader:
        batch = batch.to(device)
        metadata_x = batch.metadata_features if (use_metadata and hasattr(batch, 'metadata_features')) else None
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None

        # GAT forward restituisce: logits, emb
        _, emb = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, metadata_x=metadata_x)
        sample_embeddings_list.append(emb.cpu().numpy())

sample_embeddings = np.vstack(sample_embeddings_list)
print(f"Extracted embeddings shape: {sample_embeddings.shape}")

# Scalatura Feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample_embeddings)

# ---------------------------------------------------------
# 3. Calcolo dello Silhouette Score Quantitativo
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("SILHOUETTE SCORE ANALYSIS")
print("=" * 50)

# A) Silhouette Score per Tipo di Tessuto (tissue_type)
mask_tissue = df_merged['tissue_type'].notna()
score_tissue = silhouette_score(X_scaled[mask_tissue], df_merged.loc[mask_tissue, 'tissue_type'])
print(f"Silhouette Score (Tissue Type): {score_tissue:.4f}")

# B) Silhouette Score per Target Clinico (chemo_refractory_target)
# Nota: Si escludono i campioni senza target clinico (NaN)
mask_clinical = df_merged['chemo_refractory_target'].notna()
if mask_clinical.sum() > 0 and df_merged.loc[mask_clinical, 'chemo_refractory_target'].nunique() > 1:
    score_clinical = silhouette_score(
        X_scaled[mask_clinical],
        df_merged.loc[mask_clinical, 'chemo_refractory_target'].astype(str)
    )
    print(f"Silhouette Score (Chemo-Refractory Target): {score_clinical:.4f}")
else:
    score_clinical = None
    print("Silhouette Score (Chemo-Refractory Target): N/A (Insuffatienti campioni clinici)")

# C) Silhouette Score per Paziente (solo pazienti con almeno 2 campioni)
patient_counts = df_merged['patient_id'].value_counts()
multi_sample_pts = patient_counts[patient_counts > 1].index
mask_patient = df_merged['patient_id'].isin(multi_sample_pts)

if mask_patient.sum() > 0 and df_merged.loc[mask_patient, 'patient_id'].nunique() > 1:
    score_patient = silhouette_score(X_scaled[mask_patient], df_merged.loc[mask_patient, 'patient_id'])
    print(f"Silhouette Score (Patient ID - Multi-sample pts): {score_patient:.4f}")
else:
    score_patient = None
print("=" * 50 + "\n")

# ---------------------------------------------------------
# 4. Riduzione della Dimensionalità (PCA e UMAP)
# ---------------------------------------------------------
# A) PCA
pca = PCA(n_components=2, random_state=42)
pca_res = pca.fit_transform(X_scaled)
df_merged['PCA1'] = pca_res[:, 0]
df_merged['PCA2'] = pca_res[:, 1]

# B) UMAP
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_res = reducer.fit_transform(X_scaled)
df_merged['UMAP1'] = umap_res[:, 0]
df_merged['UMAP2'] = umap_res[:, 1]

# ---------------------------------------------------------
# 5. Visualizzazione Grafica con Silhouette in Titolo
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Subplot 1: PCA - Tipo di Tessuto
sns.scatterplot(
    data=df_merged, x='PCA1', y='PCA2', hue='tissue_type',
    palette='Set1', alpha=0.8, ax=axes[0, 0]
)
axes[0, 0].set_title(
    f'PCA by Tissue Type\n(Var. Exp: {pca.explained_variance_ratio_[:2].sum():.1%} | Silh: {score_tissue:.3f})')

# Subplot 2: PCA - Target Clinico
sns.scatterplot(
    data=df_merged, x='PCA1', y='PCA2', hue='chemo_refractory_target',
    palette={0.0: 'blue', 1.0: 'red'}, alpha=0.8, ax=axes[0, 1]
)
title_clin = f'PCA by Chemo-Refractory Target\n(Silh: {score_clinical:.3f})' if score_clinical is not None else 'PCA by Chemo-Refractory Target'
axes[0, 1].set_title(title_clin)

# Subplot 3: PCA - Primi N Pazienti (Intra-paziente)
top_patients = df_merged['patient_id'].value_counts().head(8).index
df_top_pts = df_merged[df_merged['patient_id'].isin(top_patients)]
sns.scatterplot(
    data=df_top_pts, x='PCA1', y='PCA2', hue='patient_id',
    palette='tab10', s=100, ax=axes[0, 2]
)
axes[0, 2].set_title('PCA Intra-Patient Similarity (Top 8 Patients)')

# Subplot 4: UMAP - Tipo di Tessuto
sns.scatterplot(
    data=df_merged, x='UMAP1', y='UMAP2', hue='tissue_type',
    palette='Set1', alpha=0.8, ax=axes[1, 0]
)
axes[1, 0].set_title(f'UMAP by Tissue Type\n(Silh: {score_tissue:.3f})')

# Subplot 5: UMAP - Target Clinico
sns.scatterplot(
    data=df_merged, x='UMAP1', y='UMAP2', hue='chemo_refractory_target',
    palette={0.0: 'blue', 1.0: 'red'}, alpha=0.8, ax=axes[1, 1]
)
axes[1, 1].set_title(title_clin)

# Subplot 6: UMAP - Primi N Pazienti
sns.scatterplot(
    data=df_top_pts, x='UMAP1', y='UMAP2', hue='patient_id',
    palette='tab10', s=100, ax=axes[1, 2]
)
axes[1, 2].set_title('UMAP Intra-Patient Similarity (Top 8 Patients)')

plt.tight_layout()
plt.savefig('samples_dimensionality_reduction_analysis.png', dpi=300)
plt.show()