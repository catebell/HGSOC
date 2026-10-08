import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):

  def __init__(self, in_features, hidden_dim=64):
    super().__init__()

    self.attention = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )

  def forward(self, h):
    # h shape: [N_samples, in_features]
    attn_scores = self.attention(h)  # [N_samples, 1]
    attn_scores = torch.transpose(attn_scores, 1, 0)  # [1, N_samples]
    attn_weights = F.softmax(attn_scores, dim=1)  # [1, N_samples]

    patient_emb = torch.mm(attn_weights, h)  # [1, in_features]
    return patient_emb, attn_weights


class PatientMILModel(nn.Module):

  def __init__(self, sample_encoder, num_classes=2, num_clinical_features=0,):
    super().__init__()
    self.sample_encoder = sample_encoder  # pre-trained model_samples_encoder for samples

    emb_dim = sample_encoder.emb_dim

    self.pooling = AttentionPooling(in_features=emb_dim, hidden_dim=64)

    total_dim = emb_dim + num_clinical_features
    self.classifier = nn.Linear(total_dim, num_classes)

  def forward(self, patient_batch):
    """ patient_batch: single PyG Batch object grouping every sample from each patient. """

    # pre-trained model_samples_encoder outputs class_score and sample embedding
    metadata_x = getattr(patient_batch, 'metadata_x', None)
    _, samples_embs = self.sample_encoder(
        patient_batch.x, patient_batch.edge_index, patient_batch.batch, metadata_x=metadata_x
    )  # Shape: [N_samples, emb_dim]

    # Attention Pooling aggregation
    patient_emb, attn_weights = self.pooling(samples_embs)  # Shape: [1, emb_dim], [1, N_samples]

    # eventual clinical features concatenation
    if hasattr(patient_batch, 'clinical_x') and patient_batch.clinical_x is not None:
      patient_emb = torch.cat([patient_emb, patient_batch.clinical_x], dim=1)  # [1, emb_dim + num_clinical_features]

    logits = self.classifier(patient_emb)  # final patient prediction, Shape: [1, num_classes_samples]

    return logits, attn_weights