import torch
import torch.nn as nn


class SampleMLP(nn.Module):
    """
    MLP Sample Encoder da usare in alternativa al GAT per la baseline ABMIL.
    Converte le feature geniche/omiche del tessuto in un embedding di dimensione hidden_dim.
    """

    def __init__(self, num_node_features, num_unique_genes, num_classes, hidden_channels=128, dropout=0.5,
                 num_sample_features=0):
        super(SampleMLP, self).__init__()

        # Input totale = (numero di geni * num_feature_per_gene) + eventuali metadati del campione
        self.input_dim = (num_node_features * num_unique_genes) + num_sample_features

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(256, hidden_channels),  # hidden_channels = 128 (pari all'output del GAT)
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.emb_dim = hidden_channels

        # Classificatore per la Fase 1
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_nodes, edge_index=None, batch_idx=None, metadata_x=None):
        """
        Firma compatibile sia con PyTorch Geometric / GAT che con chiamate standard.
        edge_index viene accettato per compatibilità di firma ma ignorato dall'MLP.
        """
        # Se viene chiamata come model(x, batch) con solo 2 argomenti posizionali:
        if batch_idx is None:
            if edge_index is not None and edge_index.ndim == 1:
                batch_idx = edge_index
            else:
                batch_idx = torch.zeros(x_nodes.size(0), dtype=torch.long, device=x_nodes.device)

        num_graphs = batch_idx.max().item() + 1
        num_genes = x_nodes.size(0) // num_graphs

        # Flattening del grafo in un vettore per campione: [Batch_Size, Num_Genes * Num_Features]
        x_flat = x_nodes.view(num_graphs, num_genes * x_nodes.size(1))

        if metadata_x is not None:
            x_flat = torch.cat([x_flat, metadata_x], dim=1)

        embeddings = self.encoder(x_flat)  # [Batch_Size, hidden_channels]
        logits = self.classifier(embeddings)  # [Batch_Size, num_classes]

        return logits, embeddings