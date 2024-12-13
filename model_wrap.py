from model.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, GAT
from models.rgat import RGAT

# E_GCL_mask class extends E_GCL to include a mask for coordinates
class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Written by: https://github.com/vgsatorras/egnn/blob/main/qm9/models.py

    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        pooling=False,
    ):
        """
        Forward pass for the E_GCL_mask layer.

        Args:
            h (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            coord (Tensor): Node coordinates.
            node_mask (Tensor): Node masks.
            edge_mask (Tensor): Edge masks.
            edge_attr (Tensor, optional): Edge attributes.
            node_attr (Tensor, optional): Node attributes.
            n_nodes (int, optional): Number of nodes.
            batch (Tensor, optional): Batch indices.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Updated node features, coordinates, and edge attributes.
        """
        # Initialize the parent class E_GCL
        E_GCL.__init__(
            self,
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_attr_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
            pooling=pooling,
        )

        # Remove the coordinate MLP from the parent class
        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        # Calculate the transformation for coordinates
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        # Aggregate the transformations
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        # Update coordinates with the aggregated transformations
        coord += agg * self.coords_weight
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        node_mask,
        edge_mask,
        edge_attr=None,
        node_attr=None,
        n_nodes=None,
        batch=None,
    ):
        # Compute radial and coordinate differences
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # Compute edge features
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        # Apply edge mask to edge features
        edge_feat = edge_feat * edge_mask

        # Update node features
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

# EGNN class defines an E(3)-equivariant graph neural network
class EGNN(nn.Module):
    """
    Defines an E(3)-equivariant graph neural network, which maintains equivariance
    to 3D rotations and translations.

    Attributes:
        in_node_nf (int): Number of input node features.
        in_edge_nf (int): Number of input edge features.
        hidden_nf (int): Number of hidden units.
        device (str): Device to run the model on ('cpu' or 'cuda').
        act_fn (nn.Module): Activation function.
        n_layers (int): Number of layers in the network.
        coords_weight (float): Weight for coordinate updates.
        attention (bool): Whether to use attention mechanism.
        node_attr (int): Whether to use node attributes.
    """
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=False,
        node_attr=1,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder: Linear layer to embed node features
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        # Add E_GCL layers
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_att_dim=n_node_attr,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention,
                ),
            )

        # Node decoder
        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        # Graph decoder
        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1),
        )
        self.to(self.device)

    def forward(self, h0, x, edge_index, edge_attr):
        """
        Forward pass for the EGNN model.

        Args:
            h0 (Tensor): Initial node features.
            x (Tensor): Node coordinates.
            edge_index (Tensor): Edge indices.
            edge_attr (Tensor): Edge attributes.

        Returns:
            Tuple[Tensor, Tensor]: Updated node features and coordinates.
        """
        # Embed initial node features
        h = self.embedding(h0)
        # Pass through E_GCL layers
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, x, _ = self._modules["gcl_%d" % i](
                    h,
                    edge_index,
                    x,
                    edge_attr=edge_attr,
                    node_attr=h0,
                )
            else:
                h, x, _ = self._modules["gcl_%d" % i](
                    h,
                    edge_index,
                    x,
                    edge_attr=edge_attr,
                    node_attr=None,
                )

        return h, x

# E3Pooling class for pooling operations in E(3)-equivariant networks
class E3Pooling(nn.Module):
    """
    Implements pooling operations for E(3)-equivariant networks, allowing for
    aggregation of node features in a graph.

    Attributes:
        in_node_nf (int): Number of input node features.
        in_edge_nf (int): Number of input edge features.
        hidden_nf (int): Number of hidden units.
        act_fn (nn.Module): Activation function.
        coords_weight (float): Weight for coordinate updates.
        attention (bool): Whether to use attention mechanism.
        node_attr (int): Whether to use node attributes.
        model_type (str): Type of model ('egnn' or others).
    """
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        coords_weight=1.0,
        attention=False,
        node_attr=1,
        model_type="egnn",
    ):
        super().__init__()
        self.hidden_dim = hidden_nf
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        self.model_type = model_type

        # Initialize E_GCL backbone if model type is 'egnn'
        if self.model_type == "egnn":
            self.e3_backbone = E_GCL(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                edges_in_d=in_edge_nf,
                nodes_att_dim=n_node_attr,
                act_fn=act_fn,
                recurrent=True,
                coords_weight=coords_weight,
                attention=attention,
                pooling=True,
            )

        # Use global mean pooling
        self.pool = global_mean_pool

    def forward(self, h, batch, edge_index=None, x=None, edge_attr=None):
        """
        Forward pass for the E3Pooling layer.

        Args:
            h (Tensor): Node features.
            batch (Tensor): Batch indices.
            edge_index (Tensor, optional): Edge indices.
            x (Tensor, optional): Node coordinates.
            edge_attr (Tensor, optional): Edge attributes.

        Returns:
            Tensor: Pooled node features.
        """
        # Apply E_GCL backbone if model type is 'egnn'
        if self.model_type == "egnn":
            h = self.e3_backbone(h, edge_index, x, edge_attr=edge_attr)

        # Pool the node features
        # If h is (B, N, d), take the mean across atoms
        p = self.pool(h, batch)

        return p

# FuncGNN class for functional graph neural networks
class FuncGNN(nn.Module):
    """
    Defines functional graph neural network (FuncE GNN) for multi-task learning, capable of
    handling different types of graph neural network architectures.

    Attributes:
        num_layers (int): Number of layers in the network.
        feature_dim (int): Dimension of node features.
        edge_dim (int): Dimension of edge features.
        hidden_dim (int): Number of hidden units.
        task_embed_dim (int): Dimension of task embeddings.
        num_tasks (int): Number of tasks.
        position_dim (int): Dimension of position features.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate.
        model_type (str): Type of model ('egnn', 'gat', 'rgat').
        lora_dim (int): Dimension for LoRA (Low-Rank Adaptation).
    """
    def __init__(
        self,
        num_layers,
        feature_dim,
        edge_dim,
        hidden_dim,
        task_embed_dim,
        num_tasks,
        position_dim=3,
        num_classes=3,
        dropout=0.1,
        model_type="egnn",
        lora_dim=0
    ):
        super().__init__()

        self.T = num_tasks
        self.C = num_classes
        self.model_type = model_type

        # Initialize spatial model based on model type
        if self.model_type == "egnn":
            spatial_layers = [
                EGNN(in_node_nf=feature_dim, in_edge_nf=edge_dim, hidden_nf=hidden_dim)
            ]
            for _ in range(num_layers - 1):
                spatial_layers.append(
                    EGNN(
                        in_node_nf=hidden_dim, in_edge_nf=edge_dim, hidden_nf=hidden_dim
                    )
                )
            self.spatial_model = nn.ModuleList(spatial_layers)
        elif self.model_type == "gat":
            self.spatial_model = GAT(
                in_channels=feature_dim + position_dim,
                hidden_channels=hidden_dim,
                num_layers=num_layers,
                out_channels=hidden_dim,
                dropout=dropout,
            )
        elif self.model_type == 'rgat':
            self.spatial_model = RGAT(
                in_channels=feature_dim + position_dim,
                hidden_channels=hidden_dim,
                num_layers=num_layers,
                out_channels=hidden_dim,
                dropout=dropout,
                num_relations=16,
                lora_dim=lora_dim
            )
        else:
            raise Exception("Not implemented!")

       # Initialize pooling layer
        self.pooling = E3Pooling(feature_dim, edge_dim, hidden_dim, model_type=model_type)

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + task_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Task embedding layer
        self.tasks_embed = nn.Embedding(num_tasks, task_embed_dim)

    def forward(self, h, x, edge_index, edge_attr, batch, tasks_indices, batch_size,edge_type):
        """
        Forward pass for the FuncGNN model.

        Args:
            h (Tensor): Node features.
            x (Tensor): Node coordinates.
            edge_index (Tensor): Edge indices.
            edge_attr (Tensor): Edge attributes.
            batch (Tensor): Batch indices.
            tasks_indices (Tensor): Task indices.
            batch_size (int): Batch size.
            edge_type (Tensor): Edge types.

        Returns:
            Dict[int, Tensor]: Predictions for each protein in the batch.
        """
        # Apply E(3)-equivariant layers
        if self.model_type == "egnn":
            for egnn in self.spatial_model:
                h, x = egnn(h, x, edge_index, edge_attr)

        elif self.model_type == "gat":
            input = torch.cat((h, x), dim=-1)
            h = self.spatial_model(
                x=input, edge_index=edge_index, batch=batch, batch_size=batch_size
            )
        elif self.model_type == "rgat":
            input = torch.cat((h, x), dim=-1)
            h = self.spatial_model(
                x=input, edge_index=edge_index, batch=batch, batch_size=batch_size, edge_type=edge_type, edge_attr=edge_attr
            )
        else:
            raise Exception("Not implemented!")

        # Apply E(3)-invariant pooling layer to get pooled message of shape (B, hidden_dim)
        p = self.pooling(h, batch, edge_index=edge_index, x=x, edge_attr=edge_attr)

        # Extract task indices and unique protein indices
        protein_idxs = tasks_indices[:, 0]
        unique_protein_idxs = torch.unique(protein_idxs)
        task_idxs = tasks_indices[:, 1]

        # Prepare output dictionary
        # maps protein idx b ->pred[b] (prediction for protein b)
        out = {}
        for b in range(batch_size):
            protein_idx = unique_protein_idxs[b]
            mask = protein_idxs == protein_idx
            # task indices corresponding to protein protein_idx
            task_idxs_for_protein = task_idxs[mask]
            # get the task embeddings for these indices
            task_embeddings_for_protein = self.tasks_embed(task_idxs_for_protein)

            # concatenate them to the prediction for protein protein_idx
            # need to tile the prediction for protein_idx (len of task_idxs_for_protein times)
            # (hidden_dim) -> (num_tasks_for_protein, hidden_dim)
            P = p[b].repeat(len(task_idxs_for_protein), 1)

            # Concatenate P (num_tasks_for_protein, hidden_dim) to task_embeddings_for_protein (num_tasks_for_protein, task_embed_dim)
            # to get (num_tasks_for_protein, hidden_dim + task_embed_dim)
            PT = torch.cat((P, task_embeddings_for_protein), dim=-1)
            y_pred = self.mlp(PT)
            out[b] = y_pred

        return out