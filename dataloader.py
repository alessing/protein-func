import torch
from torch_geometric.data import InMemoryDataset, download_url, Dataset, Data
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import glob
import os
import h5py
from tqdm import tqdm

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
    # Generate a random graph using the Erdős-Rényi model
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # Check if the graph is connected and non-bipartite
    if not nx.is_connected(G):
        # Find all connected components
        components = list(nx.connected_components(G))

        # Add edges between the components to connect them
        for i in range(len(components) - 1):
            # Take one node from the current component and one from the next
            node_from_comp1 = next(iter(components[i]))
            node_from_comp2 = next(iter(components[i + 1]))

            # Add an edge between them
            G.add_edge(node_from_comp1, node_from_comp2)

    # Get edges as a list of tuples (node1, node2)
    edges = list(G.edges)

    # Convert edges to edge_index format for PyTorch Geometric
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def create_fake_dataloader(num_proteins=100, num_tasks=1000):
    """
    Creates a fake dataloader for testing purposes.

    Parameters:
        num_proteins (int): Number of protein data samples to generate.
        num_tasks (int): Number of tasks for each protein.

    Returns:
        protein_datas (list): List of generated protein data.
        DataLoader: PyTorch DataLoader for the generated data.
    """
    protein_datas = []
    for i in range(num_proteins):
        # Randomly determine the number of atoms
        num_atoms = int(np.random.randint(100, 200))
        edge_indices = generate_random_graph(num_atoms, 2.0 / num_atoms)

        # Generate random atom types and structure features
        atom_types = torch.randint(1, 31, (num_atoms,), dtype=torch.long)
        structure_features = torch.rand((num_atoms, 10))

        # Generate random task indices and labels
        task_indices = torch.randint(0, num_tasks, (num_tasks, 2), dtype=torch.long)
        labels = torch.randint_like(task_indices, low=0, high=2)

        # Generate random edge features and confidence score
        edge_feats = torch.rand((3, edge_indices.shape[1])).float()
        conf_score = torch.rand((1,)).float()

        # Set the first column of labels and task indices to the protein index
        labels[:, 0] = i
        task_indices[:, 0] = i

        # Generate random positions for atoms
        pos = torch.rand((num_atoms, 3))

        # Create a Data object for each protein
        d = Data(
            edge_index=edge_indices,
            atom_types=atom_types,
            structure_features=structure_features,
            task_indices=task_indices,
            labels=labels,
            pos=pos,
            edge_attr=edge_feats.T,
            conf_score=conf_score,
        )

        protein_datas.append(d)

    return protein_datas, DataLoader(protein_datas, batch_size=4)

# Mapping of atom types to indices
atom_key = {5: 0, 6: 1, 7: 2, 15: 3}
def create_one_hot_atom_embs(in_tensor):
    """
    Converts atom types to one-hot encoded embeddings.

    Parameters:
        in_tensor (torch.Tensor): Tensor of atom types.

    Returns:
        torch.Tensor: One-hot encoded tensor of atom types.
    """
    keys = torch.tensor(list(atom_key.keys()))
    values = torch.tensor(list(atom_key.values()))

    # Map the original tensor using indexing
    new_ints = values[torch.searchsorted(keys, in_tensor)]

    # Create one-hot encoded tensor
    res = torch.nn.functional.one_hot(new_ints, num_classes=len(atom_key)).float()

    return res



def load_protein(prot_num, filename, edge_types, struct_feat_scaling=True, debug_mode=False, one_hot_atoms=True):
    """
    Loads protein data from an HDF5 file and returns a Data object.

    Parameters:
        prot_num (int): Protein number (index).
        filename (str): Path to the HDF5 file.
        edge_types (dict): Dictionary of edge types.
        struct_feat_scaling (bool): Whether to scale structure features.
        debug_mode (bool): If True, use debug mode for edge features.
        one_hot_atoms (bool): If True, convert atom types to one-hot encoding.

    Returns:
        Data: PyTorch Geometric Data object for the protein.
    """
    with h5py.File(filename, "r") as f:
         # Load data from the HDF5 file
        pos = torch.tensor(f["pos"][:])
        atom_type = torch.tensor(f["atom_type"][:], dtype=torch.long)
        structure_feats = torch.tensor(f["adj_feats"][:])
        edge_index = torch.tensor(f["edge_index"][:], dtype=torch.long)
        task_index = torch.tensor(f["task_index"][:], dtype=torch.long)
        labels = torch.tensor(f["labels"][:], dtype=torch.long)
        edge_feats = torch.tensor(f["edge_feats"]).float()
        conf_score = torch.tensor(f["confidence_score"][0]).float() / 100.

        # Convert atom types to one-hot encoding if specified
        atom_type = create_one_hot_atom_embs(atom_type)
        
        # Ensure labels and task indices have the same shape
        assert labels.shape == task_index.shape

        # Set protein number in task indices and labels
        prot_num = torch.full_like(task_index, prot_num)
        task_index = torch.stack((prot_num, task_index), dim=1)
        labels = torch.stack((prot_num, labels), dim=1)

        # Prepare edge features
        edge_feats = torch.vstack((torch.zeros((1, edge_feats.shape[1])), edge_feats))
        if debug_mode:
            edge_feats[0, :] = 0  # If debug, all edges have the same relation type
        else:
            for i, edge_type in enumerate(edge_types.keys()):
                edge_type = torch.Tensor(edge_type)
                edge_mask = (edge_feats[1:3].T == edge_type).all(dim=1)
                edge_feats[0, edge_mask] = i + 1e-6
        edge_feats = edge_feats[[0, 3]]

        # Scale structure features if specified
        if struct_feat_scaling:
            structure_feats = (1/5)*torch.minimum(structure_feats, torch.tensor(64)) + torch.log(1 + structure_feats)

        # Create a Data object for the protein
        d = Data(
            edge_index=edge_index,
            atom_types=atom_type,
            structure_features=structure_feats,
            task_indices=task_index,
            labels=labels,
            pos=pos,
            edge_attr=edge_feats[1],
            edge_type=edge_feats[0].long(),
            conf_score=conf_score,
        )
        return d


def get_dataset(dataset_dir,struct_feat_scaling=True, debug_mode=False):
    """
    Loads a dataset of proteins from a directory of HDF5 files.

    Parameters:
        dataset_dir (str): Directory containing HDF5 files.
        struct_feat_scaling (bool): Whether to scale structure features.
        debug_mode (bool): If True, use debug mode for edge features.

    Returns:
        list: List of Data objects for each protein.
        dict: Dictionary of edge types.
    """
    dataset = []

    # Define edge types
    edge_types = {
        (5, 5): 0,
        (5, 6): 1,
        (5, 7): 2,
        (5, 15): 3,
        (6, 5): 4,
        (6, 6): 5,
        (6, 7): 6,
        (6, 15): 7,
        (7, 5): 8,
        (7, 6): 9,
        (7, 7): 10,
        (7, 15): 11,
        (15, 5): 12,
        (15, 6): 13,
        (15, 7): 14,
        (15, 15): 15,
    }

    # Load each protein file and append to the dataset
    for i, fname in tqdm(enumerate(glob.glob(os.path.join(dataset_dir, "*.hdf5")))):
        d = load_protein(i, fname, edge_types, struct_feat_scaling=struct_feat_scaling, debug_mode=debug_mode)
        dataset.append(d)

    # Print edge types for reference
    print(edge_types)

    return dataset, edge_types


def get_dataloader(dataset_dir, batch_size=16, struct_feat_scaling='log', debug_mode=False):
    """
    Creates a DataLoader for a dataset of proteins.

    Parameters:
        dataset_dir (str): Directory containing HDF5 files.
        batch_size (int): Batch size for the DataLoader.
        struct_feat_scaling (str): Method for scaling structure features.
        debug_mode (bool): If True, use debug mode for edge features.

    Returns:
        list: List of Data objects for each protein.
        DataLoader: PyTorch DataLoader for the dataset.
        dict: Dictionary of edge types.
    """
    dataset, edge_types = get_dataset(dataset_dir, struct_feat_scaling, debug_mode=debug_mode)
    return dataset, DataLoader(dataset, batch_size), edge_types


if __name__ == "__main__":
    # Example usage: Load a dataloader and print each batch
    _, dl = get_dataloader("data/test_dataset")
    for d in dl:
        print(d)
