import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, AttentionalAggregation, GraphNorm


# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_nodes, gene_emb_dim=4, hidden_channels=64, heads=2, dropout=0.3, num_sample_features=0, num_edge_features=0):
        super(GAT, self).__init__()

        self.num_nodes = num_nodes
        self.num_sample_features = num_sample_features

        self.gene_embedding = torch.nn.Embedding(num_nodes, gene_emb_dim)  # Gene Identity Embedding
        in_dim = num_node_features + gene_emb_dim

        edge_dim = num_edge_features if num_edge_features > 0 else None

        # if hidden_channels dims change we need to project to map num_node_features to the output dimension before adding skip connections:
        self.skip_conn_projection1 = torch.nn.Linear(in_dim, hidden_channels * heads)
        self.conv1 = GATv2Conv(in_channels=in_dim, out_channels=hidden_channels, edge_dim=edge_dim, heads=heads)
        self.norm1 = GraphNorm(hidden_channels * heads)

        # (no need if hidden_channels is the same):
        self.skip_conn_projection2 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        self.conv2 = GATv2Conv(in_channels=hidden_channels * heads, out_channels=hidden_channels, edge_dim=edge_dim, heads=1)
        self.norm2 = GraphNorm(hidden_channels)

        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )
        self.readout = AttentionalAggregation(gate_nn=gate_nn)

        base_emb_dim = hidden_channels# * 2
        self.emb_dim = base_emb_dim + num_sample_features

        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.emb_dim, num_classes)


    def forward(self, x, edge_index, batch, edge_attr=None, metadata_x=None):
        # Gene Identity injection, [0, 1, ..., num_nodes-1] index for every graph in the batch
        node_indices = torch.arange(x.size(0), device=x.device) % self.num_nodes
        gene_emb = self.gene_embedding(node_indices)
        x = torch.cat([x, gene_emb], dim=-1)

        x_projected = self.skip_conn_projection1(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = x + x_projected
        x = F.elu(x)

        x_projected = self.skip_conn_projection2(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = x + x_projected
        x = F.elu(x)

        #x_mean = global_mean_pool(x, batch)
        #x_max = global_max_pool(x, batch)
        #emb = torch.cat([x_mean, x_max], dim=1)  #now x has dimension hidden_channels * 2

        emb = self.readout(x, batch)  # Output shape: [batch_size, hidden_channels]

        if metadata_x is not None and self.num_sample_features > 0:
            emb = torch.cat([emb, metadata_x], dim=1)

        x = self.dropout(emb)

        return self.classifier(x), emb
