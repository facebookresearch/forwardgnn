# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch_geometric.nn import SAGEConv, GATConv, GCNConv

from gnn.cached_gcn_conv import CachedGCNConv
from gnn.cached_sage_conv import CachedSAGEConv


class GNNConv(torch.nn.Module):
    def __init__(self, gnn, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if "SAGE".lower() in gnn.lower():
            if "Cached".lower() in gnn.lower():
                self.gnn = CachedSAGEConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")
            else:
                self.gnn = SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")
        elif "GCN".lower() in gnn.lower():
            if "Cached".lower() in gnn.lower():
                self.gnn = CachedGCNConv(in_channels=in_channels, out_channels=out_channels)
            else:
                self.gnn = GCNConv(in_channels=in_channels, out_channels=out_channels)
        elif "GAT".lower() in gnn.lower():
            heads = 4
            assert out_channels % heads == 0, (out_channels, heads)
            self.gnn = GATConv(in_channels=in_channels, out_channels=out_channels // heads, heads=heads)
        else:
            raise ValueError(f"Unavailable gnn: {gnn}")

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        return self.relu(self.gnn(x, edge_index))
