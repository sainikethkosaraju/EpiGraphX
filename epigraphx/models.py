"""Model zoo. Every model returns **logits** of shape [N, n_out].

Static models see one snapshot. Temporal models see a window of snapshots and
encode each with a GNN, then run a recurrent network over each node's
embedding sequence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv


class MLP(nn.Module):
    """No message passing: a check on how much the graph actually helps."""
    temporal = False

    def __init__(self, in_dim, hidden=64, n_out=1, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, x, edge_index):
        return self.net(x)


class _TwoLayerGNN(nn.Module):
    temporal = False
    conv = None

    def __init__(self, in_dim, hidden=64, n_out=1, dropout=0.3, **kw):
        super().__init__()
        self.c1 = self.make_conv(in_dim, hidden, **kw)
        self.c2 = self.make_conv(hidden, hidden, **kw)
        self.dropout = dropout
        self.out = nn.Linear(hidden, n_out)

    def make_conv(self, i, o, **kw):
        return self.conv(i, o)

    def embed(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.c2(h, edge_index))
        return F.dropout(h, self.dropout, self.training)

    def forward(self, x, edge_index):
        return self.out(self.embed(x, edge_index))


class GCN(_TwoLayerGNN):
    conv = GCNConv


class GraphSAGE(_TwoLayerGNN):
    conv = SAGEConv


class GAT(_TwoLayerGNN):
    def make_conv(self, i, o, heads=4):
        return GATConv(i, o // heads, heads=heads)


class GraphTransformer(_TwoLayerGNN):
    """Attention over neighbours (TransformerConv, Shi et al. 2021)."""
    def make_conv(self, i, o, heads=4):
        return TransformerConv(i, o // heads, heads=heads)


class TemporalGNN(nn.Module):
    """Shared GNN encoder per snapshot -> GRU/LSTM over time -> one head per output.

    ``encoder='gcn'`` with a GRU is the GCN-GRU model; ``encoder='transformer'``
    gives a temporal graph transformer; with an LSTM and several outputs it is
    the multi-horizon spatio-temporal LSTM (ST-LSTM).
    """
    temporal = True
    _encoders = {"gcn": GCN, "sage": GraphSAGE, "transformer": GraphTransformer}

    def __init__(self, in_dim, hidden=64, n_out=1, dropout=0.3, encoder="gcn", rnn="gru"):
        super().__init__()
        self.encoder = self._encoders[encoder](in_dim, hidden, 1, dropout)
        self.rnn = (nn.GRU if rnn == "gru" else nn.LSTM)(hidden, hidden, batch_first=True)
        self.dropout = dropout
        self.out = nn.Linear(hidden, n_out)

    def forward(self, xs, edge_indices):
        H = torch.stack([self.encoder.embed(x, e) for x, e in zip(xs, edge_indices)], dim=1)
        seq, _ = self.rnn(H)                                   # [N, L, hidden]
        return self.out(F.dropout(seq[:, -1], self.dropout, self.training))


def build_model(name: str, in_dim: int, hidden: int, n_out: int, dropout: float) -> nn.Module:
    static = {"MLP": MLP, "GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT,
              "GraphTransformer": GraphTransformer}
    if name in static:
        return static[name](in_dim, hidden, n_out, dropout)
    temporal = {"GCN-GRU": ("gcn", "gru"), "SAGE-GRU": ("sage", "gru"),
                "Transformer-GRU": ("transformer", "gru"), "ST-LSTM": ("sage", "lstm")}
    enc, rnn = temporal[name]
    return TemporalGNN(in_dim, hidden, n_out, dropout, encoder=enc, rnn=rnn)
