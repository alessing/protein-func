# TODO: put full RGAT here by stacking RGAT Conv Layers
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear


class LoRAGATConv(nn.Module):
    def __init__(
        self, shared_linear, in_channels, out_channels, heads, concat, dropout
    ):
        super().__init__()

        # Shared linear
        self.shared_linear = shared_linear
        # GATConv
        self.gatconv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
        )

    def forward(self, x_dict, edge_index_dict):
        x_r = self.shared_linear(x_dict, edge_index_dict)
        x_shared = self.gatconv(x_dict, edge_index_dict)

        return x_r + x_shared


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        heads,
        concat,
        dropout,
        edge_types,
    ):
        super().__init__()

        # Shared for each layer of HeteroConv
        self.shared_linears = [
            nn.Linear(in_channels, out_channels) for _ in range(num_layers)
        ]

        self.convs = torch.nn.ModuleList()
        for l in range(num_layers):
            hetero_conv_dict = {}
            for edge_type in edge_types:
                hetero_conv_dict[edge_type] = LoRAGATConv(
                    shared_linear=self.shared_linear_layers[l],
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                )

            conv = HeteroConv(hetero_conv_dict, aggr="sum")
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        # return self.lin(x_dict["author"])
        return x_dict
