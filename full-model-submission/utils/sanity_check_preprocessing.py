import lmdb
import pickle
import numpy as np

BASE = "/home/bdldt_team007/DLDockingBenchSeminar"

env = lmdb.open(f"{BASE}/data/processed/valid.lmdb", subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
txn = env.begin()
keys = sorted(list(txn.cursor().iternext(values=False)), key=lambda k: int(k.decode()))

# check first 3 entries
for k in keys[:3]:
    data = pickle.loads(txn.get(k))
    print(f"\n=== {data['pocket']} ===")
    print(f"  Ligand atoms: {len(data['atoms'])}")
    print(f"  Ligand SMILES: {data['smi'][:60]}")
    print(f"  Holo (crystal) coords range: x=[{data['holo_coordinates'][0][:,0].min():.2f}, {data['holo_coordinates'][0][:,0].max():.2f}]")
    print(f"  Pocket atoms count: {len(data['pocket_atoms'])}")
    pocket_coords = data['pocket_coordinates'][0]
    print(f"  Pocket coords range: x=[{pocket_coords[:,0].min():.2f}, {pocket_coords[:,0].max():.2f}]")

    # check distance between ligand center and pocket center -- should be small (pocket surrounds ligand)
    ligand_center = data['holo_coordinates'][0].mean(axis=0)
    pocket_center = pocket_coords.mean(axis=0)
    dist = np.linalg.norm(ligand_center - pocket_center)
    print(f"  Distance between ligand center and pocket center: {dist:.2f} Å (should be small, <5Å typically)")

    # check min distance from each pocket atom to ligand -- should mostly be <= 6Å (our cutoff)
    lig_coords = data['holo_coordinates'][0]
    min_dists = np.linalg.norm(pocket_coords[:, None, :] - lig_coords[None, :, :], axis=-1).min(axis=1)
    print(f"  Pocket atom distances to ligand: min={min_dists.min():.2f}, max={min_dists.max():.2f}, mean={min_dists.mean():.2f}")
    print(f"  (max should be close to 6.0 since that's our cutoff)")

# Now check ligand similarity across the dataset
print("\n\n=== LIGAND SIMILARITY CHECK ===")
all_smi = []
all_pockets = []
for k in keys:
    data = pickle.loads(txn.get(k))
    all_smi.append(data['smi'])
    all_pockets.append(data['pocket'])

unique_smi = len(set(all_smi))
print(f"Total complexes: {len(all_smi)}")
print(f"Unique SMILES: {unique_smi}")
print(f"Duplicate ratio: {1 - unique_smi/len(all_smi):.2%}")

# count how many complexes share the same PDBID prefix (same protein, different ligand pose)
pdb_prefixes = [p.split('_')[0] for p in all_pockets]
from collections import Counter
prefix_counts = Counter(pdb_prefixes)
print(f"\nUnique PDB prefixes (proteins): {len(prefix_counts)}")
print(f"Complexes per protein (top 10): {prefix_counts.most_common(10)}")
