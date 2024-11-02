import torch
from torch_geometric.data import InMemoryDataset, download_url, Dataset, Data
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import glob
import os
import h5py

np.random.seed(42)


def generate_random_graph(num_nodes, edge_prob):
    """
    Generates a random connected graph with no bipartite components and returns:
    - A list of edges where each edge is represented by a tuple of node indices
    - The edge_index in PyTorch Geometric format

    Parameters:
        num_nodes (int): The number of nodes in the graph.
        edge_prob (float): The probability of an edge between any two nodes (Erdős-Rényi model).

    Returns:
        edges (list of tuples): A list of edges where each edge is (node1, node2).
        edge_index (torch.Tensor): Edge index in PyTorch Geometric format (2 x num_edges).
    """
    # Step 1: Generate a random connected graph with no bipartite components
    # Generate a random graph using Erdős-Rényi model
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # Check if the graph is connected and non-bipartite
    if not nx.is_connected(G):
        # Step 2: Find all connected components
        components = list(nx.connected_components(G))

        # Step 3: Add edges between the components to connect them
        for i in range(len(components) - 1):
            # Take one node from the current component and one from the next
            node_from_comp1 = next(iter(components[i]))
            node_from_comp2 = next(iter(components[i + 1]))

            # Add an edge between them
            G.add_edge(node_from_comp1, node_from_comp2)

    # Step 2: Get edges as a list of tuples (node1, node2)
    edges = list(G.edges)

    # Step 3: Convert edges to edge_index format for PyTorch Geometric
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def create_fake_dataloader(num_proteins=100, num_tasks=1000):

    protein_datas = []
    for i in range(num_proteins):
        # print(i)
        num_atoms = int(np.random.randint(100, 200))
        edge_indices = generate_random_graph(num_atoms, 2.0 / num_atoms)

        atom_types = torch.randint(1, 31, (num_atoms,), dtype=torch.long)
        structure_features = torch.rand((num_atoms, 10))

        # num_tasks = int(np.random.exponential(10.0)) + 1
        task_indices = torch.randint(0, num_tasks, (num_tasks, 2), dtype=torch.long)
        # task_indices = torch.randint(1, num_functio, (num_tasks, 2), dtype=torch.long)
        labels = torch.randint_like(task_indices, low=0, high=2)

        labels[:, 0] = i
        task_indices[:, 0] = i

        pos = torch.rand((num_atoms, 3))

        d = Data(
            edge_index=edge_indices,
            atom_types=atom_types,
            structure_features=structure_features,
            task_indices=task_indices,
            labels=labels,
            pos=pos,
        )

        protein_datas.append(d)

    return protein_datas, DataLoader(protein_datas, batch_size=4)

def load_protein(prot_num, filename):
    with h5py.File(filename, 'r') as f:
        pos = torch.tensor(f['pos'][:])
        atom_type = torch.tensor(f['atom_type'][:], dtype=torch.long)
        structure_feats = torch.tensor(f['adj_feats'][:])
        edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)
        task_index = torch.tensor(f['task_index'][:], dtype=torch.long)
        labels = torch.tensor(f['labels'][:], dtype=torch.long)

        assert labels.shape == task_index.shape
        prot_num = torch.full_like(task_index, prot_num)
        task_index = torch.stack((task_index, prot_num), dim=1)
        labels = torch.stack((prot_num, labels), dim=1)

        d = Data(edge_index=edge_index,
                atom_types=atom_type,
                structure_features=structure_feats,
                task_indices=task_index,
                labels=labels,
                pos=pos)
        return d

def get_dataset(dataset_dir):
    dataset = []

    for i, fname in enumerate(glob.glob(os.path.join(dataset_dir, '*.hdf5'))):
        dataset.append(load_protein(i, fname))

    return dataset

def get_dataloader(dataset_dir, batch_size=16):
    dataset = get_dataset(dataset_dir)
    return dataset, DataLoader(dataset, batch_size)


if __name__ == '__main__':
    _, dl = get_dataloader('data/test_dataset')
    for d in dl:
        print(d)
