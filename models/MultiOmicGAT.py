import torch
import torch.nn.functional as F

from torch.nn import Dropout, Module, Linear, Sequential, ReLU
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GraphNorm


# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html
# solves static attention problem mentioned in the original paper

class MultiOmicGAT(Module):
    def __init__(self, in_node_features, hidden_dim, out_channels, heads=2, dropout=0.2, num_sample_features=0):
        super(MultiOmicGAT, self).__init__()

        # Layer 1: receives node feats (RNA + CNV) and applies multi-head attention
        self.gat1 = GATv2Conv(
            in_channels=in_node_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,  # --> dim = hidden_dim * heads
            add_self_loops=True,
            dropout=dropout,
        )

        # linear projection for residual connection (Skip Connection)
        self.residual_proj1 = Linear(in_node_features, hidden_dim * heads)
        # Norm Layer to stabilize training
        self.norm1 = GraphNorm(hidden_dim * heads)

        # Layer 2: features compression and genes relation consolidation
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=True,
            dropout=dropout,
        )

        self.residual_proj2 = Linear(hidden_dim * heads, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)

        self.num_sample_features = num_sample_features

        if num_sample_features > 0:
            self.sample_metadata_encoder = Sequential(
                Linear(num_sample_features, 32),
                ReLU(),
                Dropout(p=dropout)
            )

            fusion_dim = hidden_dim * 2 + 32
            total_emb_dim = fusion_dim
        else:
            total_emb_dim = hidden_dim * 2

        # graph classification/prediction
        self.classifier = Sequential(
            Linear(total_emb_dim, hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, out_channels)
        )

        self.dropout_rate = dropout
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch=None, metadata_x=None):
        """
        x: Tensor [N_nodes, in_node_features] -> Genes features matrix
        edge_index: Tensor [2, E] -> Adjacency matrix
        batch: Tensor [N_nodes] -> to combine more graphs, observe them together
        """

        # --- LAYER 1 ---
        h_res = self.residual_proj1(x)  # Residual connection
        h = self.gat1(x, edge_index)
        h = self.norm1(h)
        h = h + h_res  # TODO pre-norm vs post-norm?
        h = F.elu(h)  # ELU activation (GAT standard)

        # --- LAYER 2 ---
        h_res = self.residual_proj2(h)
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = h + h_res  # TODO pre-norm vs post-norm?
        h = F.elu(h)

        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # [Batch_Size, hidden_dim * 2]

        graph_emb = self.dropout(graph_emb)

        if self.num_sample_features > 0 and metadata_x is not None:
            metadata_emb = self.sample_metadata_encoder(metadata_x)
            fused_emb = torch.cat([graph_emb, metadata_emb], dim=1) # [Batch_Size, hidden_dim * 2 + 32]
            out = self.classifier(fused_emb)
            return out, fused_emb
        else:
            out = self.classifier(graph_emb)
            return out, graph_emb
