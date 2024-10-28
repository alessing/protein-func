import h5py
import numpy as np


def save_hdf5(filename='test.hdf5'):

    num_atoms = int(np.random.randint(low=200, high=50000))
    num_tasks = int(np.random.exponential(10.)) + 1

    pos = np.random.uniform(-1, 1, (num_atoms, 3))
    atom_type = np.random.randint(low=1, high=30, size=(num_atoms,))
    adj_feats = np.random.uniform(low=0.1, high=20, size=(num_atoms,))
    edge_index = np.random.randint(low=0, high=num_atoms, size=(2, num_atoms))
    task_index = np.random.randint(low=0, high=20000, size=(num_tasks))
    labels = np.random.randint(low=0, high=20000, size=(num_tasks))
    

    with open(filename, 'w') as f:
        f.create_dataset('pos', data=pos)
        f.create_dataset('atom_type', data=atom_type)
        f.create_dataset('adj_feats', data=adj_feats)
        f.create_dataset('edge_index', data=edge_index)
        f.create_dataset('task_index', data=task_index)
        f.create_dataset('labels', data=labels)

if __name__ == '__main__':
    save_hdf5()
