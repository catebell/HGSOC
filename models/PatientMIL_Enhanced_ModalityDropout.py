import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax

import config_file_tree

""" AttentionPooling + MaxPooling, Modality Dropout on node features """


class DualPathAttentionPooling(nn.Module):
    """
    Combina Attention Pooling (significato globale del bag) e Max Pooling
    (rilevamento di campioni o cloni resistenti estremi).
    """

    def __init__(self, in_features, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, h, bag_index):
        # 1. Attention Weighted Mean Pooling
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        gated_h = self.dropout(a_v * a_u)

        attn_scores = self.attention_weights(gated_h).squeeze(-1)
        attn_weights = softmax(attn_scores, bag_index)

        weighted_h = h * attn_weights.unsqueeze(-1)
        attn_patient_emb = global_add_pool(weighted_h, bag_index)

        # 2. Max Pooling (cattura il picco di resistenza)
        max_patient_emb = global_max_pool(h, bag_index)

        # Concatenazione delle due prospettive [Mean_Attn || Max]
        patient_emb = torch.cat([attn_patient_emb, max_patient_emb], dim=-1)
        return patient_emb, attn_weights


class PatientMILModel(nn.Module):
    def __init__(
            self,
            sample_encoder,
            num_classes=2,
            num_tissue_classes=3,
            tissue_emb_dim=16,
            proj_dim=32,
            pooling_hidden_dim=64,
            classifier_hidden_dim=32,
            num_clinical_features=0,
            dropout=0.3,
            subsample_rate=0.85,
            modality_dropout_prob=0.20,
    ):
        super().__init__()
        self.sample_encoder = sample_encoder
        self.tissue_emb_dim = tissue_emb_dim
        self.subsample_rate = subsample_rate
        self.modality_dropout_prob = modality_dropout_prob

        if tissue_emb_dim > 0:
            self.tissue_embedding = nn.Embedding(num_tissue_classes, tissue_emb_dim)
        else:
            self.tissue_embedding = None

        sample_emb_dim = getattr(sample_encoder, 'emb_dim', 64)
        total_sample_dim = sample_emb_dim + (tissue_emb_dim if tissue_emb_dim > 0 else 0)

        # 1. Sample Projection Layer: adatta le feature GAT al task MIL del paziente
        if config_file_tree.PATIENT_TARGET_COL == 'chemo_refractory_target':
            self.sample_projector = nn.Sequential(
                nn.Linear(in_features=total_sample_dim, out_features=16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Dropout(p=0.4)
            )
        else:
            self.sample_projector = nn.Sequential(
                nn.Linear(total_sample_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.pooling = DualPathAttentionPooling(in_features=proj_dim, hidden_dim=pooling_hidden_dim, dropout=dropout)
        pooled_dim = proj_dim * 2

        # 3. Normalizzazione separata per feature cliniche e di bag
        self.bag_norm = nn.LayerNorm(pooled_dim)

        if num_clinical_features > 0:
            self.clinical_norm = nn.LayerNorm(num_clinical_features)
        else:
            self.clinical_norm = None

        total_patient_dim = pooled_dim + num_clinical_features

        # 4. Classificatore finale compatto
        if config_file_tree.PATIENT_TARGET_COL == 'chemo_refractory_target':
            self.classifier = nn.Sequential(
                nn.Linear(in_features=total_patient_dim, out_features=16),
                nn.GELU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=16, out_features=num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(total_patient_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes)
            )

    def forward(self, patient_batch):
        # ----------------------------------------------------
        # STRATEGIA CNV: Modality Dropout sulle Feature di Nodo
        # ----------------------------------------------------
        x = patient_batch.x.clone()

        # Viene applicato SOLO durante la fase di training
        if self.training and self.modality_dropout_prob > 0.0:
            # Con probabilità = modality_dropout_prob, azzera temporaneamente l'espressione genica (indice 0)
            if torch.rand(1).item() < self.modality_dropout_prob:
                x[:, 0] = 0.0  # Mantiene attiva SOLO la feature di CNV (log2CopyNumber, indice 1)
        # ----------------------------------------------------

        # A. Feature Extraction GAT
        metadata_x = getattr(patient_batch, 'metadata_x', None)
        _, samples_embs = self.sample_encoder(
            x,
            patient_batch.edge_index,
            patient_batch.batch,
            metadata_x=metadata_x
        )

        # B. Tissue Embedding
        if self.tissue_embedding is not None and hasattr(patient_batch, 'y') and patient_batch.y is not None:
            y_tissue = patient_batch.y.long()
            tissue_embs = self.tissue_embedding(y_tissue)
            samples_embs = torch.cat([samples_embs, tissue_embs], dim=-1)

        # C. Proiezione nello spazio latente del paziente
        projected_samples = self.sample_projector(samples_embs)

        # D. Bag Index
        if hasattr(patient_batch, 'num_samples') and patient_batch.num_samples is not None:
            num_samples_per_patient = patient_batch.num_samples
            bag_index = torch.repeat_interleave(
                torch.arange(len(num_samples_per_patient), device=samples_embs.device),
                num_samples_per_patient
            )
        else:
            bag_index = torch.zeros(samples_embs.size(0), dtype=torch.long, device=samples_embs.device)

        # E. Dual-Path Pooling (Attention + Max)
        patient_emb, attn_weights = self.pooling(projected_samples, bag_index)
        patient_emb = self.bag_norm(patient_emb)

        # F. Fusione regolata delle Feature Cliniche
        if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
            clinical_x = patient_batch.clinical_x
            if clinical_x.ndim == 1:
                clinical_x = clinical_x.unsqueeze(0)
            clinical_x = self.clinical_norm(clinical_x)
            patient_emb = torch.cat([patient_emb, clinical_x], dim=1)

        logits = self.classifier(patient_emb)
        return logits, attn_weights