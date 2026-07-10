import lmdb, pickle, numpy as np

BASE = '/home/bdldt_team007/DLDockingBenchSeminar'
env = lmdb.open(f'{BASE}/data/processed/train.lmdb', subdir=False, readonly=True)

with env.begin() as txn:
    total = txn.stat()['entries']
    print(f"Total entries: {total}")
    
    # check first 5 entries
    for i in range(min(5, total)):
        sample = pickle.loads(txn.get(str(i).encode()))
        coords = np.array(sample['coordinates'][0])
        pocket_coords = np.array(sample['pocket_coordinates'][0])
        print(f"\nEntry {i} ({sample['pocket']}):")
        print(f"  ligand atoms: {len(sample['atoms'])}, coords shape: {coords.shape}")
        print(f"  pocket atoms: {len(sample['pocket_atoms'])}, coords shape: {pocket_coords.shape}")
        print(f"  ligand coord range: {coords.min():.2f} to {coords.max():.2f}")
        print(f"  pocket coord range: {pocket_coords.min():.2f} to {pocket_coords.max():.2f}")
        print(f"  any NaN in ligand: {np.any(np.isnan(coords))}")
        print(f"  any NaN in pocket: {np.any(np.isnan(pocket_coords))}")
