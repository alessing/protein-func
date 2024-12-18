import h5py
import numpy as np
import pandas as pd
import os
import ast
from Bio import PDB
from tqdm import tqdm
import torch
import argparse

# Define directories for raw and processed data

RAW_DATA = "data/raw_data"
ALPHA_FOLD_DIR = "UP000005640_9606_HUMAN_v4"
PROCESSED_DATA = "data/processed_data"

# Mapping of elements to their periodic table indices
PERIODIC_TABLE_IDX = {
    'H': 0,    'He': 1,   'Li': 2,   'Be': 3,   'B': 4,    'C': 5,    'N': 6,    'O': 7,    'F': 8,    'Ne': 9,
    'Na': 10,  'Mg': 11,  'Al': 12,  'Si': 13,  'P': 14,   'S': 15,   'Cl': 16,  'Ar': 17,  'K': 18,   'Ca': 19,
    'Sc': 20,  'Ti': 21,  'V': 22,   'Cr': 23,  'Mn': 24,  'Fe': 25,  'Co': 26,  'Ni': 27,  'Cu': 28,  'Zn': 29,
    'Ga': 30,  'Ge': 31,  'As': 32,  'Se': 33,  'Br': 34,  'Kr': 35,  'Rb': 36,  'Sr': 37,  'Y': 38,   'Zr': 39,
    'Nb': 40,  'Mo': 41,  'Tc': 42,  'Ru': 43,  'Rh': 44,  'Pd': 45,  'Ag': 46,  'Cd': 47,  'In': 48,  'Sn': 49,
    'Sb': 50,  'Te': 51,  'I': 52,   'Xe': 53,  'Cs': 54,  'Ba': 55,  'La': 56,  'Ce': 57,  'Pr': 58,  'Nd': 59,
    'Pm': 60,  'Sm': 61,  'Eu': 62,  'Gd': 63,  'Tb': 64,  'Dy': 65,  'Ho': 66,  'Er': 67,  'Tm': 68,  'Yb': 69,
    'Lu': 70,  'Hf': 71,  'Ta': 72,  'W': 73,   'Re': 74,  'Os': 75,  'Ir': 76,  'Pt': 77,  'Au': 78,  'Hg': 79,
    'Tl': 80,  'Pb': 81,  'Bi': 82,  'Po': 83,  'At': 84,  'Rn': 85,  'Fr': 86,  'Ra': 87,  'Ac': 88,  'Th': 89,
    'Pa': 90,  'U': 91,   'Np': 92,  'Pu': 93,  'Am': 94,  'Cm': 95,  'Bk': 96,  'Cf': 97,  'Es': 98,  'Fm': 99,
    'Md': 100, 'No': 101, 'Lr': 102, 'Rf': 103, 'Db': 104, 'Sg': 105, 'Bh': 106, 'Hs': 107, 'Mt': 108, 'Ds': 109,
    'Rg': 110, 'Cn': 111, 'Nh': 112, 'Fl': 113, 'Mc': 114, 'Lv': 115, 'Ts': 116, 'Og': 117
}

# Determine the device to use for torch operations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def adjacency_features(adj_matrix, D):
    """
    Calculate adjacency features for a given adjacency matrix.
    
    Parameters:
    adj_matrix (torch.Tensor): The adjacency matrix.
    D (int): The number of features to compute.
    
    Returns:
    torch.Tensor: The computed structure features.
    """
    try:
        adj_matrix = adj_matrix.to(device=DEVICE, dtype=torch.float32)
        structure_features = torch.ones((adj_matrix.shape[0], D), device=DEVICE)
    except:
        adj_matrix = adj_matrix.to(device="cpu", dtype=torch.float32)
        structure_features = torch.ones((adj_matrix.shape[0], D), device="cpu")
    Apow = adj_matrix
    for d in range(1, D):
        structure_features[:, d] = torch.diag(Apow.to_dense())
        if d < D-1:
            Apow = adj_matrix @ Apow
        torch.cuda.empty_cache()

    adj_matrix = adj_matrix.to("cpu")

    return structure_features

def save_hdf5(filepath, protein_funcs, parser, D):
    """
    Save protein data to an HDF5 file.
    
    Parameters:
    filepath (str): Path to the protein file.
    protein_funcs (pd.DataFrame): DataFrame containing protein functions.
    parser (PDBParser): PDB parser object.
    D (int): Number of features to compute.
    """
    uniprot_id = filepath.split("-")[1]
    protein_data = protein_funcs[protein_funcs["DB_Object_ID"] == uniprot_id]
    structure = parser.get_structure(uniprot_id, filepath)
    
    pos = []
    atom_type = []
    pLDDTs = []  # List to store confidence scores for residues (https://alphafold.ebi.ac.uk/faq)
    for res in structure.get_residues():
        pLDDT = 0  # Initialize confidence score for residue
        for atom in res:
            pLDDT = atom.get_bfactor()  # All atoms in a residue have the same pLDDT
            pos.append(atom.get_coord())
            atom_type.append(PERIODIC_TABLE_IDX[atom.element])
        pLDDTs.append(pLDDT)
    pos = torch.tensor(np.array(pos), device=DEVICE, dtype=torch.float16)
    atom_type = np.array(atom_type)
    confidence_score = sum(pLDDTs) / len(pLDDTs) # Average confidence score
    
    # Calculate pairwise distances between atoms
    pos_diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distances = torch.sqrt(torch.sum(pos_diffs * pos_diffs, axis=2))
    del pos_diffs

    # Create adjacency matrix based on distance threshold
    adj_matrix = torch.where(distances < 3, 1.0, 0.0)
    adj_matrix.fill_diagonal_(0.0)  # Mask out self edges, which has dist 0
    edge_index_tensor = torch.nonzero(torch.triu(adj_matrix))  # Mask out lower triangle, and get i and j of edges
    edge_index = edge_index_tensor.T.cpu().numpy()

    # Calculate edge distances and features
    edge_dists = distances * torch.where(distances < 3, 1.0, 0.0)
    edge_dists = edge_dists[edge_index[0], edge_index[1]]
    edge_dists = edge_dists.cpu()
    edge_feats = np.vstack((atom_type[edge_index], edge_dists))  # row 1 is atom type 1, row 2 is atom type 2, row 3 is dist
    del distances
    del edge_dists

    # Compute adjacency node features
    adj_node_feats = adjacency_features(adj_matrix.to_sparse(), D)
    del adj_matrix

    # Extract task index and labels from protein data
    task_index = ast.literal_eval(protein_data["GO_Idx"].iloc[0])
    labels = ast.literal_eval(protein_data["Qualifier_Idx"].iloc[0])
    
    # Save data to HDF5 file
    with h5py.File(f"{PROCESSED_DATA}/hdf5_files_d_{D}/{uniprot_id}.hdf5", 'w') as f:
        f.create_dataset('pos', data=pos.cpu().numpy())
        f.create_dataset('atom_type', data=atom_type)
        f.create_dataset('confidence_score', data=[confidence_score])
        f.create_dataset('adj_feats', data=adj_node_feats.cpu().numpy())
        f.create_dataset('edge_feats', data=edge_feats)
        f.create_dataset('edge_index', data=edge_index)
        f.create_dataset('task_index', data=task_index)
        f.create_dataset('labels', data=labels)

def main():
    """
    Main function to parse arguments and process protein data.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-d",
        type=int,
        default=10
    )
    args = arg_parser.parse_args()

    parser = PDB.PDBParser()
    protein_funcs = pd.read_csv(f"{PROCESSED_DATA}/protein_functions.csv")
    for protein_id in tqdm(protein_funcs["DB_Object_ID"]):
        protein_path = f"{RAW_DATA}/{ALPHA_FOLD_DIR}/AF-{protein_id}-F1-model_v4.pdb"
        if os.path.isfile(protein_path):
            save_hdf5(protein_path, protein_funcs, parser, args.d)

if __name__ == '__main__':
    main()