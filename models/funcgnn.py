# Written by: https://github.com/vgsatorras/egnn/blob/main/qm9/models.py

from model.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, GAT
from models.rgat import RGAT
from torch_scatter import scatter
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation
import math


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
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

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
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
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
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

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
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

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1),
        )
        self.to(self.device)

    def forward(self, h0, x, edge_index, edge_attr):
        h = self.embedding(h0)
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


# Adapted from: https://projects.volkamerlab.org/teachopencadd/talktorials/T036_e3_equivariant_gnn.html
class EquivariantMPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        act: nn.Module,
        batch_norm: bool,
        dropout: float,
        edge_types: torch.Tensor,
        rank: int,
        device="cpu",
    ) -> None:
        super().__init__()
        self.act = act
        self.residual_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        self.edge_types = edge_types
        self.rank = rank
        self.hidden_channels = hidden_channels
        self.device = device
        self.use_batch_norm = batch_norm

        # Messages will consist of two (source and target) node embeddings and a scalar distance
        self.message_input_size = 2 * in_channels + 1

        self.weight_dict = {}

        # edge_types is a dict from (source_atom_type, target_atom_type) to edge_type_idx
        for edge_type_idx in edge_types.values():
            # A, B matrix for each edge type r
            # Each will have bias because it's an r-dim vector, which is small
            A_r = nn.Linear(self.rank, self.hidden_channels, bias=True)
            B_r = nn.Linear(self.message_input_size, self.rank, bias=False)

            nn.init.kaiming_uniform_(A_r.weight, a=math.sqrt(5))
            B_r.weight.data.fill_(0.0)

            self.weight_dict[f"{edge_type_idx}_A"] = A_r
            self.weight_dict[f"{edge_type_idx}_B"] = B_r

        self.weight_dict = nn.ModuleDict(self.weight_dict)

        # No bias
        self.W_shared = nn.Linear(
            self.message_input_size, self.hidden_channels, bias=False
        )

        # equation (4) "psi_l" NN
        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + self.hidden_channels, self.hidden_channels),
            act,
        )

        self.dropout = nn.Dropout(p=dropout)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_channels)
        self.activation = nn.ReLU()

    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def node_message_function(
        self,
        source_node_embed,  # h_i
        target_node_embed,  # h_j
        edge_feat,
    ):
        # implements equation (3)
        # atom_1_type, atom_2_type, node_dist = edge_feat[:, 0], edge_feat[:, 1], edge_feat[:, 2]
        edge_type_idxs, node_dist = edge_feat[:, 0], edge_feat[:, 1]
        num_edges = edge_feat.shape[0]

        unique_edge_type_idxs = torch.unique(edge_type_idxs)

        m_agg = torch.zeros(
            (num_edges, self.hidden_channels),
            device=self.device,
            dtype=source_node_embed.dtype,
        )
        for edge_type_idx in unique_edge_type_idxs:
            # A_r, B_r = self.weight_dict[int(edge_type_idx.item())]
            A_r = self.weight_dict[f"{int(edge_type_idx.item())}_A"]
            B_r = self.weight_dict[f"{int(edge_type_idx.item())}_B"]

            # for all edges in atom graph of protein, find edges with edge_type_idx
            mask = edge_type_idxs == edge_type_idx

            # only use the source and target embeddings (and dist) for the edges with edge_idx
            source_node_embed_edge_type = source_node_embed[mask, :]
            target_node_embed_edge_type = target_node_embed[mask, :]
            node_dist_edge_type = node_dist[mask]
            message_repr = torch.cat(
                (
                    source_node_embed_edge_type,
                    target_node_embed_edge_type,
                    node_dist_edge_type.unsqueeze(-1),
                ),
                dim=-1,
            )

            # Apply LoRA in a compositional manner
            m_r = A_r(B_r(message_repr)) + self.W_shared(message_repr)

            # place the messages corresponding to edge_type_idx in appropriate place
            m_agg[mask, :] = m_r

        return m_agg

    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, node_embed, node_pos, edge_index, edge_attr):
        row, col = edge_index

        # compute messages "m_ij" from  equation (3)
        node_messages = self.node_message_function(
            node_embed[row], node_embed[col], edge_feat=edge_attr
        )

        # message sum aggregation in equation (4)
        aggr_node_messages = scatter(node_messages, col, dim=0, reduce="mean")

        # batchnorm
        if self.use_batch_norm:
            aggr_node_messages = self.batch_norm(aggr_node_messages)

        # dropout
        new_node_embed = self.dropout(aggr_node_messages)

        # activation
        new_node_embed = self.activation(new_node_embed)

        # compute new node embeddings "h_i^{l+1}"
        # (implements rest of equation (4))
        new_node_embed = self.residual_proj(node_embed) + self.node_update_mlp(
            torch.cat((node_embed, aggr_node_messages), dim=-1)
        )

        return new_node_embed


class EquivariantGNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        final_embedding_size=None,
        target_size: int = 1,
        num_mp_layers: int = 2,
        rank: int = 16,
        dropout: float = 0.5,
        batch_norm: bool = True,
        edge_types=None,
        device="cpu",
    ) -> None:
        super().__init__()
        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        # non-linear activation func.
        # usually configurable, here we just use Relu for simplicity
        self.act = nn.ReLU()

        # equation (1) "psi_0"
        # self.f_initial_embed = nn.Embedding(100, hidden_channels)
        self.f_initial_embed = nn.Linear(input_channels, hidden_channels)

        # create stack of message passing layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [final_embedding_size]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            layer = EquivariantMPLayer(
                d_in, d_out, self.act, dropout, batch_norm, edge_types, rank, device
            )
            self.message_passing_layers.append(layer)

        # modules required for readout of a graph-level
        # representation and graph-level property prediction
        self.aggregation = MeanAggregation()  # SumAggregation()
        self.f_predict = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

    def encode(self, x, h, edge_index, edge_attr):
        # theory, equation (1)
        node_embed = self.f_initial_embed(h)
        # message passing
        # theory, equation (3-4)
        for mp_layer in self.message_passing_layers:
            # NOTE here we use the complete edge index defined by the transform earlier on
            # to implement the sum over $j \neq i$ in equation (4)
            node_embed = mp_layer(node_embed, x, edge_index, edge_attr)

        return node_embed

    def _predict(self, node_embed, batch_index):
        aggr = self.aggregation(node_embed, batch_index)
        return self.f_predict(aggr)

    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x, h, edge_index, edge_attr, batch):
        node_embed = self.encode(x, h, edge_index, edge_attr)
        # pred = self._predict(node_embed, batch)
        # return pred
        return node_embed


class E3Pooling(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        coords_weight=1.0,
        attention=False,
        node_attr=1,
        model_type="egnn_t0",
    ):
        super().__init__()
        self.hidden_dim = hidden_nf
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        self.model_type = model_type

        if self.model_type == "egnn_old":
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

        self.pool = global_mean_pool

    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, h, batch, edge_index=None, x=None, edge_attr=None):
        if self.model_type == "egnn_old":
            h = self.e3_backbone(h, edge_index, x, edge_attr=edge_attr)

        # If h is (B, N, d), take the mean across atoms
        # p = h.mean(dim=1)
        p = self.pool(h, batch)

        return p


class FuncGNN(nn.Module):
    def __init__(
        self,
        num_layers,
        feature_dim,
        edge_dim,
        hidden_dim,
        task_embed_dim,
        num_tasks,
        edge_types,
        rank=16,
        position_dim=3,
        num_classes=3,
        dropout=0.4,
        batch_norm=True,
        device="cpu",
        model_type="egnn_t0",
    ):
        super().__init__()

        self.T = num_tasks
        self.C = num_classes
        self.model_type = model_type
        self.edge_types = edge_types

        if self.model_type == "egnn_old":
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
        elif self.model_type == "egnn_t0":
            self.spatial_model = EquivariantGNN(
                input_channels=feature_dim,
                hidden_channels=hidden_dim,
                final_embedding_size=hidden_dim,
                target_size=hidden_dim,
                num_mp_layers=num_layers,
                rank=rank,
                dropout=dropout,
                batch_norm=batch_norm,
                edge_types=edge_types,
                device=device,
            )
        elif self.model_type == "gat":
            self.spatial_model = GAT(
                in_channels=feature_dim + position_dim,
                hidden_channels=hidden_dim,
                num_layers=num_layers,
                out_channels=hidden_dim,
                dropout=dropout,
            )
        elif self.model_type == "rgat":
            # TODO: SHOULD WE STACK (H, X) as init node feature
            self.spatial_model = RGAT(
                in_channels=feature_dim,
                out_channels=hidden_dim,
                hidden_channels=hidden_dim,
                num_heads=4,
                num_relations=16,
                num_layers=num_layers,
                edge_dim=1,
                num_blocks=None,
                dropout=dropout,
            )
        else:
            raise Exception("Not implemented!")

        # self.softmax = nn.Softmax(dim=-1)
        self.pooling = E3Pooling(
            feature_dim, edge_dim, hidden_dim, model_type=model_type
        )

        # produces W[p, t] + b where p is the pooled message
        # self.mlp = nn.Linear(hidden_dim + task_embed_dim, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + task_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # task embedding
        self.tasks_embed = nn.Embedding(num_tasks, task_embed_dim)

    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, h, x, edge_index, edge_attr, batch, tasks_indices, batch_size):
        # Apply E(3)-equivariant layers
        if self.model_type == "egnn_old":
            for egnn in self.spatial_model:
                h, x = egnn(h, x, edge_index, edge_attr)

        elif self.model_type == "egnn_t0":
            h = self.spatial_model(x, h, edge_index, edge_attr, batch)

        elif self.model_type == "gat":
            input = torch.cat((h, x), dim=-1)
            h = self.spatial_model(
                x=input, edge_index=edge_index, batch=batch, batch_size=batch_size
            )
        elif self.model_type == "rgat":
            print(f"h: {h.shape}")
            breakpoint()
            edge_type = None

            h = self.spatial_model(h, edge_index, edge_type, edge_attr)

        else:
            raise Exception("Not implemented!")

        # Apply E(3)-invariant pooling layer to get pooled message
        # Shape (B, hidden_dim) or (B, 64)
        p = self.pooling(h, batch, edge_index=edge_index, x=x, edge_attr=edge_attr)

        protein_idxs = tasks_indices[:, 0]
        unique_protein_idxs = torch.unique(protein_idxs)
        task_idxs = tasks_indices[:, 1]

        # maps protein idx b ->pred[b] (prediction for protein b)
        out = {}
        # iterate over batch of proteins
        for b in range(batch_size):
            protein_idx = unique_protein_idxs[b]
            mask = protein_idxs == protein_idx

            # indices for all tasks corresponding to protein protein_idx
            tasks_indices_for_protein = task_idxs[mask]
            # get the task embeddings for these indices (i.e. for each task relevant to the protein)
            # this will give (num_tasks, task_embedding_dim)
            tasks_embedding_for_protein = self.tasks_embed(tasks_indices_for_protein)

            # concatenate them to the prediction for protein protein_idx
            # need to tile the prediction for protein_idx (len of task_idxs_for_protein times)
            # (hidden_dim) -> (num_tasks_for_protein, hidden_dim)
            P = p[b].repeat(len(tasks_indices_for_protein), 1)

            # Concatenate P (num_tasks_for_protein, hidden_dim) to tasks_embedding_for_protein (num_tasks_for_protein, task_embed_dim)
            # to get (num_tasks_for_protein, hidden_dim + task_embed_dim)
            PT = torch.cat((P, tasks_embedding_for_protein), dim=-1)

            # maps PT to (num_tasks_for_protein, num_classes)
            y_pred = self.mlp(PT)
            out[b] = y_pred

        return out
