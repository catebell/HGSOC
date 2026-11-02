import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax


class GatedAttentionPooling(nn.Module):
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
        # h shape: [total_tissues_in_batch, in_features]
        # bag_index shape: [total_tissues_in_batch]
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        gated_h = self.dropout(a_v * a_u)

        attn_scores = self.attention_weights(gated_h).squeeze(-1)
        attn_weights = softmax(attn_scores, bag_index)

        weighted_h = h * attn_weights.unsqueeze(-1)
        patient_emb = global_add_pool(weighted_h, bag_index)
        return patient_emb, attn_weights


class PatientMILModel(nn.Module):
    def __init__(
        self,
        sample_encoder,
        num_node_features=2,
        num_classes=2,
        num_tissue_classes=3,
        tissue_emb_dim=16,
        pooling_hidden_dim=64,
        classifier_hidden_dim=64,
        num_clinical_features=0,
        dropout=0.3,
        subsample_rate=0.85
    ):
        super().__init__()
        self.sample_encoder = sample_encoder
        self.subsample_rate = subsample_rate
        self.tissue_emb_dim = tissue_emb_dim

        if tissue_emb_dim > 0:
            self.tissue_embedding = nn.Embedding(num_tissue_classes, tissue_emb_dim)
        else:
            self.tissue_embedding = None

        sample_emb_dim = getattr(sample_encoder, 'emb_dim', 64)
        total_sample_dim = sample_emb_dim + (tissue_emb_dim if tissue_emb_dim > 0 else 0)

        self.pooling = GatedAttentionPooling(in_features=total_sample_dim, hidden_dim=pooling_hidden_dim, dropout=dropout)
        total_patient_dim = total_sample_dim + num_clinical_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_patient_dim),
            nn.Dropout(dropout),
            nn.Linear(total_patient_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, patient_batch):
        # A. Feature Extraction dei tessuti tramite Sample Encoder (GAT)
        metadata_x = getattr(patient_batch, 'metadata_x', None)
        _, samples_embs = self.sample_encoder(
            patient_batch.x,
            patient_batch.edge_index,
            patient_batch.batch,
            metadata_x=metadata_x
        )

        # B. Embedding del Tipo di Tessuto (se abilitato)
        if self.tissue_embedding is not None and hasattr(patient_batch, 'y') and patient_batch.y is not None:
            y_tissue = patient_batch.y.long()
            tissue_embs = self.tissue_embedding(y_tissue)
            samples_embs = torch.cat([samples_embs, tissue_embs], dim=-1)

        # C. Costruzione dinamica del bag_index (Graph-to-Patient index)
        if hasattr(patient_batch, 'num_samples') and patient_batch.num_samples is not None:
            num_samples_per_patient = patient_batch.num_samples
            bag_index = torch.repeat_interleave(
                torch.arange(len(num_samples_per_patient), device=samples_embs.device),
                num_samples_per_patient
            )
        else:
            bag_index = torch.zeros(samples_embs.size(0), dtype=torch.long, device=samples_embs.device)

        # D. Aggregazione MIL tramite Gated Attention
        patient_emb, attn_weights = self.pooling(samples_embs, bag_index)

        # E. Concatenazione Feature Cliniche (se presenti)
        if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
            clinical_x = patient_batch.clinical_x
            if clinical_x.ndim == 1:
                clinical_x = clinical_x.unsqueeze(0)
            patient_emb = torch.cat([patient_emb, clinical_x], dim=1)

        logits = self.classifier(patient_emb)
        return logits, attn_weights