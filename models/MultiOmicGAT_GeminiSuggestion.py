import torch
import torch.nn.functional as F

from torch.nn import Dropout, Module, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_max_pool, GraphNorm, AttentionalAggregation


class MultiOmicGAT(Module):
    def __init__(self, in_node_features, hidden_dim, out_channels, heads=4, dropout=0.2, num_sample_features=0):
        super(MultiOmicGAT, self).__init__()
        self.dropout_rate = dropout

        # Normalizza i 2 canali di input (RNA e CNV) al volo prima della GNN
        self.input_bn = BatchNorm1d(in_node_features)

        # 1. Encoder Iniziale dei Nodi: Mappa [RNA, CNV] -> hidden_dim
        self.node_encoder = Sequential(
            Linear(in_node_features, hidden_dim),
            ReLU(),
            Dropout(p=dropout)
        )

        # 2. Layer GAT 1
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,  # Output dim: hidden_dim * heads
            add_self_loops=True,
            dropout=dropout,
        )
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.res_proj1 = Linear(hidden_dim, hidden_dim * heads)

        # 3. Layer GAT 2
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,  # Output dim: hidden_dim
            add_self_loops=True,
            dropout=dropout,
        )
        self.norm2 = GraphNorm(hidden_dim)
        self.res_proj2 = Linear(hidden_dim * heads, hidden_dim)

        # 4. Attentive Readout/Pooling: impara quali geni sono biologicamente rilevanti
        gate_nn = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1)
        )
        self.att_pool = AttentionalAggregation(gate_nn=gate_nn)

        # Dimensione totale embedding del grafo (Attentive Pool + Max Pool = hidden_dim * 2)
        graph_emb_dim = hidden_dim * 2

        # 5. Gestione Metadati Clinici/Campione
        self.num_sample_features = num_sample_features
        if num_sample_features > 0:
            self.sample_metadata_encoder = Sequential(
                Linear(num_sample_features, 32),
                ReLU(),
                Dropout(p=dropout)
            )
            total_emb_dim = graph_emb_dim + 32
        else:
            total_emb_dim = graph_emb_dim

        # 6. Classificatore / Multi-Layer Perceptron di output
        self.classifier = Sequential(
            Linear(total_emb_dim, hidden_dim),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index, batch=None, metadata_x=None):
        if batch is None:
            print("in forward, batch is none")
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_bn(x)

        # Proiezione iniziale delle feature dei geni
        h_in = self.node_encoder(x)

        # --- LAYER 1 ---
        h_res = self.res_proj1(h_in)
        h = self.gat1(h_in, edge_index)
        h = self.norm1(h, batch)
        h = F.elu(h + h_res)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # --- LAYER 2 ---
        h_res = self.res_proj2(h)
        h = self.gat2(h, edge_index)
        h = self.norm2(h, batch)
        h = F.elu(h + h_res)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # --- GRAPH-LEVEL EMBEDDING (POOLING) ---
        att_emb = self.att_pool(h, batch)        # [Batch_Size, hidden_dim] -> Selezione ponderata dei geni
        max_emb = global_max_pool(h, batch)      # [Batch_Size, hidden_dim] -> Cattura i picchi estremi
        graph_emb = torch.cat([att_emb, max_emb], dim=1)  # [Batch_Size, hidden_dim * 2]

        # --- FUSIONE METADATI E PREDIZIONE ---
        if self.num_sample_features > 0 and metadata_x is not None:
            metadata_emb = self.sample_metadata_encoder(metadata_x)
            fused_emb = torch.cat([graph_emb, metadata_emb], dim=1)
        else:
            fused_emb = graph_emb

        out = self.classifier(fused_emb)
        return out, fused_emb