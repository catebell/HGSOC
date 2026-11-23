import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax

import config_file_tree


class DualPathAttentionPooling(nn.Module):
    """
    Attention Pooling (global bag meaning, all the samples) with Max Pooling (extreme samples).
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

        # 2. Max Pooling (captures signal peaks)
        max_patient_emb = global_max_pool(h, bag_index)

        patient_emb = torch.cat([attn_patient_emb, max_patient_emb], dim=-1)  # two POVs
        return patient_emb, attn_weights


class PatientMILModel(nn.Module):
    def __init__(
            self,
            sample_encoder,
            num_classes=2,
            num_tissue_classes=3,
            tissue_emb_dim=16,
            proj_dim=24,
            pooling_hidden_dim=16,
            classifier_hidden_dim=16,
            num_clinical_features=0,
            dropout=0.3,
            subsample_rate=1.0,
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

        # 1. Sample Projection Layer compatto e regolarizzato
        self.sample_projector = nn.Sequential(
            nn.Linear(total_sample_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Dual-Path Pooling allineato a proj_dim
        self.pooling = DualPathAttentionPooling(
            in_features=proj_dim,
            hidden_dim=pooling_hidden_dim,
            dropout=dropout
        )
        pooled_dim = proj_dim * 2

        # 3. Normalizzazione separata per feature cliniche e di bag
        self.bag_norm = nn.LayerNorm(pooled_dim)

        if num_clinical_features > 0:
            self.clinical_norm = nn.LayerNorm(num_clinical_features)
        else:
            self.clinical_norm = None

        total_patient_dim = pooled_dim + num_clinical_features

        # 4. Classificatore finale compatto
        self.classifier = nn.Sequential(
            nn.Linear(total_patient_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def freeze_sample_encoder(self):
        """Congela completamente la GAT (usato nella fase di warmup della testa MIL)."""
        for param in self.sample_encoder.parameters():
            param.requires_grad = False
        self.sample_encoder.eval()

    def unfreeze_top_gat_layers(self, layer_names=('conv2', 'norm2', 'readout')):
        """
        Sblocca solo gli strati terminali della GAT, mantenendo sempre congelati
        gene_embedding (28.7k parametri) e conv1/norm1/skip_conn_projection1.
        """
        for param in self.sample_encoder.parameters():
            param.requires_grad = False

        for name in layer_names:
            layer = getattr(self.sample_encoder, name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        """
        Garantisce che i sottomoduli congelati della GAT restino in eval()
        (disattivando dropout e aggiornamenti di statistiche) anche quando
        train_epoch_mil() chiama model.train().
        """
        super().train(mode)
        if mode:
            for module in self.sample_encoder.children():
                params = list(module.parameters())
                if params and not any(p.requires_grad for p in params):
                    module.eval()
            if hasattr(self.sample_encoder, 'conv1') and hasattr(self.sample_encoder, 'dropout'):
                if not any(p.requires_grad for p in self.sample_encoder.conv1.parameters()):
                    self.sample_encoder.dropout.eval()
        return self

    def encode_patient_bag(self, patient_batch):
        """ Extract latent patient embedding after pooling and normalization, before final classifier.
        Applies Modality Dropout and Instance Subsampling (in training). """
        x = patient_batch.x.clone()

        # Modality Dropout on node features, applied only in training phase
        if 0 in config_file_tree.NODE_FEATURE_INDEX and self.training and self.modality_dropout_prob > 0.0:
            if torch.rand(1).item() < self.modality_dropout_prob:
                x[:, 0] = 0.0  # applied on expression (main relevance)

        metadata_x = getattr(patient_batch, 'metadata_x', None)
        _, samples_embs = self.sample_encoder(
            x,
            patient_batch.edge_index,
            patient_batch.batch,
            metadata_x=metadata_x
        )

        if self.tissue_embedding is not None and hasattr(patient_batch, 'y') and patient_batch.y is not None:
            y_tissue = patient_batch.y.long()
            tissue_embs = self.tissue_embedding(y_tissue)
            samples_embs = torch.cat([samples_embs, tissue_embs], dim=-1)

        projected_samples = self.sample_projector(samples_embs)  # in patient latent space

        if hasattr(patient_batch, 'num_samples') and patient_batch.num_samples is not None:
            num_samples_per_patient = patient_batch.num_samples
            bag_index = torch.repeat_interleave(
                torch.arange(len(num_samples_per_patient), device=samples_embs.device),
                num_samples_per_patient
            )
        else:
            num_samples_per_patient = [samples_embs.size(0)]
            bag_index = torch.zeros(samples_embs.size(0), dtype=torch.long, device=samples_embs.device)

        '''
        # 2. Instance Subsampling Asimmetrico nel bag (solo in training e solo su pazienti 'no' / sensibili)
        if self.training and self.subsample_rate < 1.0:
            keep_mask = torch.rand(projected_samples.size(0), device=projected_samples.device) < self.subsample_rate

            # Preserva sempre il 100% dei campioni per i pazienti platino-resistenti (patient_y == 1)
            patient_y = getattr(patient_batch, 'patient_y', None)
            if patient_y is not None:
                patient_labels = patient_y.view(-1)
                sample_patient_labels = patient_labels[bag_index]
                keep_mask = keep_mask | (sample_patient_labels == 1)
            else:
                patient_labels = None

            # Garantisce che ogni paziente sensibile mantenga almeno 1 campione nel bag
            for p_idx in range(len(num_samples_per_patient)):
                if patient_labels is not None and patient_labels[p_idx] == 1:
                    continue
                p_sample_indices = (bag_index == p_idx).nonzero(as_tuple=True)[0]
                if p_sample_indices.numel() > 0 and not keep_mask[p_sample_indices].any():
                    rand_choice = p_sample_indices[
                        torch.randint(0, p_sample_indices.numel(), (1,), device=projected_samples.device)
                    ]
                    keep_mask[rand_choice] = True
        


            projected_samples = projected_samples[keep_mask]
            bag_index = bag_index[keep_mask]
        # tested and worse
        '''

        patient_emb, attn_weights = self.pooling(projected_samples, bag_index)
        patient_emb = self.bag_norm(patient_emb)

        if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
            clinical_x = patient_batch.clinical_x
            if clinical_x.ndim == 1:
                clinical_x = clinical_x.unsqueeze(0)
            clinical_x = self.clinical_norm(clinical_x)
            patient_emb = torch.cat([patient_emb, clinical_x], dim=1)

        return patient_emb, attn_weights

    def forward(self, patient_batch):
        patient_emb, attn_weights = self.encode_patient_bag(patient_batch)
        logits = self.classifier(patient_emb)
        return logits, attn_weights