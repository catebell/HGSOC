import torch.nn as nn


class PatientBottleneckMLP(nn.Module):
    """
    MLP compatto a singolo bottleneck per coorti piccole (N < 100).
    Il vettore z (16-D) funge da embedding supervisionato del paziente.
    """
    def __init__(self, in_dim, bottleneck_dim=16, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def encode(self, x):
        return self.encoder(x)  # Embedding paziente [N_patients, 16]

    def forward(self, x):
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z