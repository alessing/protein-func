# TODO: put full RGAT here by stacking RGAT Conv Layers

import torch
from models.rgat_conv import RGATConv
import torch.nn.functional as F


class RGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_relations,
        num_heads,
        num_layers,
        edge_dim,
        num_blocks=None,
        concat=False,
        dropout=0.6,
    ):
        super(RGAT, self).__init__()
        assert num_layers >= 2, "Number of layers must be at least 2."

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()  # Batch normalization layers

        self.convs.append(
            RGATConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                num_relations=num_relations,
                num_blocks=num_blocks,
                heads=num_heads,
                attention_mechanism="within-relation",
                attention_mode="additive-self-attention",
                concat=concat,
                edge_dim=edge_dim,  # Include edge features
                dropout=dropout,
            )
        )

        if concat:
            hidden_dim = hidden_channels * num_heads
        else:
            hidden_dim = hidden_channels

        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                    num_blocks=num_blocks,
                    heads=num_heads,
                    attention_mechanism="within-relation",
                    attention_mode="additive-self-attention",
                    concat=concat,
                    edge_dim=edge_dim,  # Include edge features
                    dropout=dropout,
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(
            RGATConv(
                in_channels=hidden_dim,
                out_channels=out_channels,
                num_relations=num_relations,
                num_blocks=num_blocks,
                heads=1,  # Single head for the final layer
                attention_mechanism="within-relation",
                attention_mode="additive-self-attention",
                concat=concat,
                edge_dim=edge_dim,  # Include edge features
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index, edge_type, edge_attr):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(
                x, edge_index, edge_type, edge_attr
            )  # Pass edge_attr to the convolution
            # x = bn(x)  # Apply batch normalization
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x = self.convs[-1](x, edge_index, edge_type, edge_attr)

        return x


# class RGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
#         super().__init__()
#         self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
#         self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
#         self.lin = torch.nn.Linear(hidden_channels, out_channels)

#     def forward(self, x, edge_index, edge_type):
#         x = self.conv1(x, edge_index, edge_type).relu()
#         x = self.conv2(x, edge_index, edge_type).relu()
#         x = self.lin(x)
#         return x


if __name__ == "__main__":
    # Parameters for the model
    in_channels = 16  # Input feature size
    hidden_channels = 32
    out_channels = 4  # Number of output classes
    num_relations = 5  # Number of edge relation types
    num_heads = 4  # Number of attention heads per relation
    num_layers = 3  # Number of RGAT layers
    edge_dim = 1  # One-dimensional edge feature

    # Create the model
    model = RGAT(
        in_channels,
        out_channels,
        hidden_channels,
        num_relations,
        num_heads,
        num_layers,
        edge_dim,
    )

    # Example data (dummy)
    x = torch.randn((10, in_channels))  # Node features (10 nodes, 16 features each)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]]
    )  # Edges (source, target pairs)
    edge_type = torch.tensor([0, 1, 2, 3])  # Edge types corresponding to relations
    edge_attr = torch.randn(
        (edge_index.size(1), edge_dim)
    )  # One-dimensional edge features

    # Forward pass
    output = model(x, edge_index, edge_type, edge_attr)
    print(output.shape)
