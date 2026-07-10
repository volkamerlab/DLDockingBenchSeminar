import lmdb, pickle
import numpy as np

def check_array(name, arr, pdbid, sample_idx, issues):
    if np.isnan(arr).any():
        issues.append(f"Sample {sample_idx} ({pdbid}): NaN found in {name}")
    if np.isinf(arr).any():
        issues.append(f"Sample {sample_idx} ({pdbid}): Inf found in {name}")

def check_min_dist(name, arr, pdbid, sample_idx, issues, threshold=1e-3):
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_d = dist.min()
    if min_d < threshold:
        n_close = (dist < threshold).sum() // 2
        issues.append(f"Sample {sample_idx} ({pdbid}): {name} has {n_close} atom pair(s) with distance < {threshold} Å (min={min_d:.6f} Å)")
    return min_d

for path, name in [
    ("/home/bdldt_team007/DLDockingBenchSeminar/data/processed/train.lmdb", "train"),
    ("/home/bdldt_team007/DLDockingBenchSeminar/data/processed/valid.lmdb", "valid"),
]:
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    issues = []
    min_dists_ligand = []
    min_dists_pocket = []

    with env.begin() as txn:
        n = env.stat()['entries']
        for i in range(n):
            entry = pickle.loads(txn.get(f"{i}".encode()))
            pdbid = entry.get('pocket', 'unknown')

            for ci, conf in enumerate(entry['coordinates']):
                check_array(f"coordinates[{ci}]", conf, pdbid, i, issues)
                d = check_min_dist(f"coordinates[{ci}]", conf, pdbid, i, issues)
                min_dists_ligand.append(d)

            for ci, arr in enumerate(entry['holo_coordinates']):
                check_array(f"holo_coordinates[{ci}]", arr, pdbid, i, issues)
                d = check_min_dist(f"holo_coordinates[{ci}]", arr, pdbid, i, issues)
                min_dists_ligand.append(d)

            for ci, arr in enumerate(entry['pocket_coordinates']):
                check_array(f"pocket_coordinates[{ci}]", arr, pdbid, i, issues)
                d = check_min_dist(f"pocket_coordinates[{ci}]", arr, pdbid, i, issues)
                min_dists_pocket.append(d)

            for ci, arr in enumerate(entry['holo_pocket_coordinates']):
                check_array(f"holo_pocket_coordinates[{ci}]", arr, pdbid, i, issues)

    min_dists_ligand = np.array(min_dists_ligand)
    min_dists_pocket = np.array(min_dists_pocket)

    print(f"\n=== {name}.lmdb : {n} samples ===")
    print(f"Ligand-side arrays checked: {len(min_dists_ligand)}")
    print(f"  min pairwise distance overall: {min_dists_ligand.min():.6f} Å")
    print(f"  arrays with min dist < 1e-3 Å: {(min_dists_ligand < 1e-3).sum()}")
    print(f"  arrays with min dist < 0.5 Å:  {(min_dists_ligand < 0.5).sum()}")
    print(f"Pocket-side arrays checked: {len(min_dists_pocket)}")
    print(f"  min pairwise distance overall: {min_dists_pocket.min():.6f} Å")
    print(f"  arrays with min dist < 1e-3 Å: {(min_dists_pocket < 1e-3).sum()}")
    print(f"  arrays with min dist < 0.5 Å:  {(min_dists_pocket < 0.5).sum()}")

    print(f"\nTotal issues found: {len(issues)}")
    for issue in issues[:30]:
        print(f"  - {issue}")
    if len(issues) > 30:
        print(f"  ... and {len(issues)-30} more")
