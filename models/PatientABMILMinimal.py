import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax

import config_file_tree


class LinearAttentionPooling(nn.Module):
    """
    Attention Pooling minimalista a singolo strato lineare (65 parametri su in_features=64).
    """

    def __init__(self, in_features=64, dropout=0.25):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = nn.Linear(in_features, 1)

    def forward(self, h, bag_index):
        h_dropped = self.dropout(h)
        attn_scores = self.attention_weights(h_dropped).squeeze(-1)  # [N_samples_in_batch]
        attn_weights = softmax(attn_scores, bag_index)               # [N_samples_in_batch]

        weighted_h = h * attn_weights.unsqueeze(-1)
        patient_emb = global_add_pool(weighted_h, bag_index)         # [N_patients, in_features]
        return patient_emb, attn_weights


class PatientABMILMinimal(nn.Module):
    """
    ABMIL minimalista per coorti piccole (es. NACT, N=78):
    - Sample Encoder (GAT) completamente congelato (64-D)
    - Rimozione a 0 parametri del bias anatomico (centratura per tissue_type sul train set)
    - Single-layer Linear Attention Pooling (65 parametri)
    - L2-normalization (0 parametri) + Linear Classifier diretto (130 parametri)
    """

    def __init__(
        self,
        sample_encoder,
        num_classes=2,
        num_tissue_classes=3,
        num_clinical_features=0,
        dropout=0.25,
        subsample_rate=0.80,
        modality_dropout_prob=0.15,
        deconfound_tissue=True,
    ):
        super().__init__()
        self.sample_encoder = sample_encoder
        self.num_tissue_classes = num_tissue_classes
        self.subsample_rate = subsample_rate
        self.modality_dropout_prob = modality_dropout_prob
        self.deconfound_tissue = deconfound_tissue

        sample_emb_dim = getattr(sample_encoder, 'emb_dim', 64)
        self.sample_emb_dim = sample_emb_dim

        # Buffer salvato nel state_dict (0 parametri addestrabili): centroidi per tissue_type
        self.register_buffer(
            'tissue_centroids',
            torch.zeros(num_tissue_classes, sample_emb_dim, dtype=torch.float32)
        )
        self.register_buffer('centroids_fitted', torch.tensor(False))

        # 1. Single-layer Linear Attention Pooling (65 parametri)
        self.pooling = LinearAttentionPooling(
            in_features=sample_emb_dim,
            dropout=dropout
        )

        # 2. Normalizzazione clinica (se usata)
        if num_clinical_features > 0:
            self.clinical_norm = nn.LayerNorm(num_clinical_features)
        else:
            self.clinical_norm = None

        total_patient_dim = sample_emb_dim + num_clinical_features

        # 3. Classificatore lineare diretto senza hidden layer (130 parametri con num_clinical_features=0)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_patient_dim, num_classes)

        # Congela sempre la GAT
        self.freeze_sample_encoder()

    def freeze_sample_encoder(self):
        """Congela interamente la GAT."""
        for param in self.sample_encoder.parameters():
            param.requires_grad = False
        self.sample_encoder.eval()

    def train(self, mode: bool = True):
        """Mantiene la GAT in eval() anche durante model.train()."""
        super().train(mode)
        self.sample_encoder.eval()
        return self

    @torch.no_grad()
    def fit_tissue_deconfounding(self, device, train_loader):
        """
        Calcola sul solo training set il centroide medio dell'embedding GAT per ciascun
        tissue_type, in modo da centrare le biopsie ed eliminare il confondente anatomico
        senza aggiungere parametri addestrabili.
        """
        if not self.deconfound_tissue:
            return

        was_training = self.training
        self.eval()

        sums = torch.zeros(self.num_tissue_classes, self.sample_emb_dim, device=device)
        counts = torch.zeros(self.num_tissue_classes, device=device)

        for patient_batch in train_loader:
            patient_batch = patient_batch.to(device)
            metadata_x = getattr(patient_batch, 'metadata_x', None)
            _, samples_embs = self.sample_encoder(
                patient_batch.x,
                patient_batch.edge_index,
                patient_batch.batch,
                metadata_x=metadata_x
            )

            if hasattr(patient_batch, 'y') and patient_batch.y is not None:
                y_tissue = patient_batch.y.long()
                for t_idx in range(self.num_tissue_classes):
                    mask = (y_tissue == t_idx)
                    if mask.any():
                        sums[t_idx] += samples_embs[mask].sum(dim=0)
                        counts[t_idx] += mask.sum()

        for t_idx in range(self.num_tissue_classes):
            if counts[t_idx] > 0:
                self.tissue_centroids[t_idx] = sums[t_idx] / counts[t_idx]
            else:
                self.tissue_centroids[t_idx].zero_()

        self.centroids_fitted.fill_(True)
        if was_training:
            self.train()

    def encode_patient_bag(self, patient_batch):
        """
        Estrae l'embedding latente del paziente z_i in R^64 (+ eventuali feature cliniche)
        e i pesi di attenzione per biopsia.
        """
        x = patient_batch.x.clone()

        # Modality Dropout leggero in training
        if 0 in config_file_tree.NODE_FEATURE_INDEX and self.training and self.modality_dropout_prob > 0.0:
            if torch.rand(1).item() < self.modality_dropout_prob:
                x[:, 0] = 0.0

        metadata_x = getattr(patient_batch, 'metadata_x', None)
        with torch.no_grad():
            _, samples_embs = self.sample_encoder(
                x,
                patient_batch.edge_index,
                patient_batch.batch,
                metadata_x=metadata_x
            )

        # De-confounding tissutale: sottrae la media del sito anatomico calcolata sul train set
        if self.deconfound_tissue and self.centroids_fitted and hasattr(patient_batch, 'y') and patient_batch.y is not None:
            y_tissue = patient_batch.y.long()
            samples_embs = samples_embs - self.tissue_centroids[y_tissue]

        if hasattr(patient_batch, 'num_samples') and patient_batch.num_samples is not None:
            num_samples_per_patient = patient_batch.num_samples
            bag_index = torch.repeat_interleave(
                torch.arange(len(num_samples_per_patient), device=samples_embs.device),
                num_samples_per_patient
            )
        else:
            num_samples_per_patient = [samples_embs.size(0)]
            bag_index = torch.zeros(samples_embs.size(0), dtype=torch.long, device=samples_embs.device)

        # Instance Subsampling nei bag in training (garantisce almeno 1 campione per paziente)
        if self.training and self.subsample_rate < 1.0:
            keep_mask = torch.rand(samples_embs.size(0), device=samples_embs.device) < self.subsample_rate
            for p_idx in range(len(num_samples_per_patient)):
                p_sample_indices = (bag_index == p_idx).nonzero(as_tuple=True)[0]
                if p_sample_indices.numel() > 0 and not keep_mask[p_sample_indices].any():
                    rand_choice = p_sample_indices[
                        torch.randint(0, p_sample_indices.numel(), (1,), device=samples_embs.device)
                    ]
                    keep_mask[rand_choice] = True

            samples_embs = samples_embs[keep_mask]
            bag_index = bag_index[keep_mask]

        # Attention Pooling pesato
        patient_emb, attn_weights = self.pooling(samples_embs, bag_index)

        # L2-normalization parameter-free (proietta i pazienti sulla ipersfera unitaria)
        patient_emb = F.normalize(patient_emb, p=2, dim=-1)

        if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
            clinical_x = patient_batch.clinical_x
            if clinical_x.ndim == 1:
                clinical_x = clinical_x.unsqueeze(0)
            clinical_x = self.clinical_norm(clinical_x)
            patient_emb = torch.cat([patient_emb, clinical_x], dim=1)

        return patient_emb, attn_weights

    def forward(self, patient_batch):
        patient_emb, attn_weights = self.encode_patient_bag(patient_batch)
        logits = self.classifier(self.head_dropout(patient_emb))
        return logits, attn_weights