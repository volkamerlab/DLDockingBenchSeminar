import lmdb, pickle
import numpy as np

env = lmdb.open(
    "/home/bdldt_team007/DLDockingBenchSeminar/data/processed/train.lmdb",
    subdir=False, readonly=True, lock=False
)

with env.begin() as txn:
    n = env.stat()['entries']
    print(f"Total samples: {n}")
    bad = []
    for i in range(n):
        entry = pickle.loads(txn.get(f"{i}".encode()))
        for ci, conf in enumerate(entry['coordinates']):
            if np.isnan(conf).any() or np.isinf(conf).any():
                bad.append((i, 'conformer', ci))
        for name in ['holo_coordinates', 'pocket_coordinates', 'holo_pocket_coordinates']:
            for ci, arr in enumerate(entry[name]):
                if np.isnan(arr).any() or np.isinf(arr).any():
                    bad.append((i, name, ci))
        # check number of atoms matches number of coordinates
        n_atoms = len(entry['atoms'])
        for ci, conf in enumerate(entry['coordinates']):
            if conf.shape[0] != n_atoms:
                bad.append((i, 'atom_count_mismatch', f"atoms={n_atoms} coords={conf.shape[0]}"))
        n_pocket_atoms = len(entry['pocket_atoms'])
        for ci, arr in enumerate(entry['pocket_coordinates']):
            if arr.shape[0] != n_pocket_atoms:
                bad.append((i, 'pocket_atom_count_mismatch', f"atoms={n_pocket_atoms} coords={arr.shape[0]}"))

    print(f"Bad samples found: {len(bad)}")
    for b in bad[:20]:
        print(b)
