# Written by: https://github.com/vgsatorras/egnn/blob/main/qm9/models.py

from model.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool


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
    ):
        super().__init__()
        self.hidden_dim = hidden_nf
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

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

    def forward(self, h, edge_index, x, edge_attr=None, batch=None):
        h = self.e3_backbone(h, edge_index, x, edge_attr=edge_attr)
        # If h is (B, N, d), take the mean across atoms
        # p = h.mean(dim=1)
        p = self.pool(h, batch)

        return p


class FuncGNN(nn.Module):
    def __init__(
        self,
        num_equivariant_layers,
        feature_dim,
        edge_dim,
        hidden_dim,
        task_embed_dim,
        num_tasks,
        num_classes=3,
    ):
        super().__init__()

        self.T = num_tasks
        self.C = num_classes

        egnns = [
            EGNN(in_node_nf=feature_dim, in_edge_nf=edge_dim, hidden_nf=hidden_dim)
        ]
        for _ in range(num_equivariant_layers - 1):
            egnns.append(
                EGNN(in_node_nf=hidden_dim, in_edge_nf=edge_dim, hidden_nf=hidden_dim)
            )
        # self.egnns = nn.ModuleList(
        #     [
        #         EGNN(in_node_nf=feature_dim, in_edge_nf=edge_dim, hidden_nf=hidden_dim)
        #         for _ in range(num_equivariant_layers)
        #     ]
        # )
        self.egnns = nn.ModuleList(egnns)

        # self.softmax = nn.Softmax(dim=-1)
        self.pooling = E3Pooling(feature_dim, edge_dim, hidden_dim)

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

        # task embedding vector (size num_tasks)
        # self.tasks_embed = nn.Parameter(torch.rand(num_tasks, task_embed_dim))
        # TODO: Use Embedding here
        self.tasks_embed = nn.Embedding(num_tasks, task_embed_dim)

    def forward(self, h, x, edge_index, edge_attr, batch, tasks_indices):
        # Apply E(3)-equivariant layers
        for egnn in self.egnns:
            h, x = egnn(h, x, edge_index, edge_attr)

        # Apply E(3)-invariant pooling layer to get pooled message
        # Shape (B, hidden_dim) or (B, 64)
        p = self.pooling(h, edge_index, x, edge_attr=edge_attr, batch=batch)
        # print(f"p: {p.shape}")

        # iterate over each p in batch
        B = p.size(0)
        # print(len(tasks_indices[:, 0]))
        # print(tasks_indices[:, 0][-10:])
        # unique_protein_idxs, inverse_idxs = torch.unique(
        #     tasks_indices[:, 0], return_inverse=True, sorted=False
        # )
        
        protein_idxs = tasks_indices[:, 0]
        unique_protein_idxs = torch.unique(protein_idxs)
        task_idxs = tasks_indices[:, 1]

        # out = torch.zeros(B, self.T, self.C)

        # maps protein idx b ->pred[b] (prediction for protein b)
        out = {}
        for b in range(B):
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
            # out[b] = y_pred
            out[b] = y_pred

            # ground truth
            # y = labels[:, 1][mask]
            # ce_loss = torch.nn.CrossEntropyLoss()
            # protein_loss = ce_loss(y_pred, y)
            # print(f"Protein loss: {protein_loss}")

        return out
