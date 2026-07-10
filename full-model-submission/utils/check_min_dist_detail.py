import lmdb, pickle
import numpy as np

env = lmdb.open(
    "/home/bdldt_team007/DLDockingBenchSeminar/data/processed/train.lmdb",
    subdir=False, readonly=True, lock=False
)

with env.begin() as txn:
    n = env.stat()['entries']
    min_dists = []
    n_atoms_list = []
    for i in range(n):
        entry = pickle.loads(txn.get(f"{i}".encode()))
        n_atoms_list.append(len(entry['atoms']))
        for conf in entry['coordinates']:
            diff = conf[:, None, :] - conf[None, :, :]
            d = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(d, 99)
            min_dists.append(d.min())

    min_dists = np.array(min_dists)
    n_atoms_list = np.array(n_atoms_list)
    print(f"Total conformers checked: {len(min_dists)}")
    print(f"Min distance overall: {min_dists.min():.4f}")
    print(f"Distances < 1.0: {(min_dists < 1.0).sum()}")
    print(f"Distances < 0.8: {(min_dists < 0.8).sum()}")
    print(f"Distances < 0.6: {(min_dists < 0.6).sum()}")
    print(f"Max atom count: {n_atoms_list.max()}, samples >100 atoms: {(n_atoms_list>100).sum()}")
    print(f"Max seq len after +2 special tokens: {n_atoms_list.max()+2}")
