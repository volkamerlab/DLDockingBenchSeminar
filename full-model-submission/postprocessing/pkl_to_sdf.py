"""
pkl_to_sdf.py — Convert infer.py output (valid.pkl) to individual {complex_id}_pred.sdf files.
"""

import os
import pickle
import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = "/home/bdldt_team007/DLDockingBenchSeminar"
PKL_FILE  = f"{BASE}/results/proto_test/valid.pkl"
LMDB_FILE = f"{BASE}/data/processed/valid.lmdb"
OUT_DIR   = f"{BASE}/results/proto_test"
CONF_SIZE = 10

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load ordered complex_ids from LMDB ────────────────────────────────────
env = lmdb.open(LMDB_FILE, subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=256)
txn = env.begin()
raw_keys = list(txn.cursor().iternext(values=False))
raw_keys_sorted = sorted(raw_keys, key=lambda k: int(k.decode()))

complex_ids = []
smiles_list = []
for k in raw_keys_sorted:
    data = pickle.loads(txn.get(k))
    complex_ids.append(data["pocket"])
    smiles_list.append(data["smi"])

print(f"Loaded {len(complex_ids)} complex_ids from LMDB")
print(f"First 3: {complex_ids[:3]}")

# ── 2. Parse valid.pkl ────────────────────────────────────────────────────────
with open(PKL_FILE, "rb") as f:
    batches = pickle.load(f)

print(f"Loaded {len(batches)} batches from pkl")

coord_predict_list  = []
holo_center_list    = []
prmsd_score_list    = []

for batch in batches:
    sz = batch["atoms"].size(0)
    for i in range(sz):
        token_mask = batch["atoms"][i] > 2
        coord_predict = batch["coord_predict"][i][token_mask].numpy().astype(np.float32)
        holo_center   = batch["holo_center_coordinates"][i][:3].numpy().astype(np.float32)
        prmsd_score   = batch["prmsd_score"][i].item()
        coord_predict_list.append(coord_predict)
        holo_center_list.append(holo_center)
        prmsd_score_list.append(prmsd_score)

total_poses = len(coord_predict_list)
n_complexes = total_poses // CONF_SIZE
print(f"Total poses: {total_poses}, complexes: {n_complexes}, conf_size: {CONF_SIZE}")

assert n_complexes == len(complex_ids), \
    f"Mismatch: {n_complexes} complexes in pkl vs {len(complex_ids)} in LMDB"

# ── 3. For each complex: pick best pose, set coords, write SDF ───────────────
success = 0
failed  = 0

for i in range(n_complexes):
    complex_id = complex_ids[i]
    smi        = smiles_list[i]

    start = i * CONF_SIZE
    end   = start + CONF_SIZE

    coords_tta  = coord_predict_list[start:end]
    scores_tta  = prmsd_score_list[start:end]
    centers_tta = holo_center_list[start:end]

    best_idx    = int(np.argmin(scores_tta))
    best_coords = coords_tta[best_idx]
    best_center = centers_tta[best_idx]

    # reconstruct heavy-atom-only molecule from SMILES
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"  [SKIP] {complex_id}: could not parse SMILES")
        failed += 1
        continue

    # remove Hs to match the heavy-atom-only coords from token_mask
    mol = Chem.RemoveHs(mol)

    # generate a conformer to get correct bond geometry / atom ordering
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # fallback: try without ETKDGv3
        result = AllChem.EmbedMolecule(mol, randomSeed=42, clearConfs=True)
    if result == -1:
        print(f"  [SKIP] {complex_id}: EmbedMolecule failed")
        failed += 1
        continue

    if best_coords.shape[0] != mol.GetNumAtoms():
        print(f"  [SKIP] {complex_id}: atom count mismatch "
              f"(coords={best_coords.shape[0]}, mol={mol.GetNumAtoms()})")
        failed += 1
        continue

    # set predicted coordinates + add back center offset
    conf = mol.GetConformer(0)
    for j in range(best_coords.shape[0]):
        x = float(best_coords[j, 0] + best_center[0])
        y = float(best_coords[j, 1] + best_center[1])
        z = float(best_coords[j, 2] + best_center[2])
        conf.SetAtomPosition(j, Chem.rdGeometry.Point3D(x, y, z))

    # write SDF
    out_path = os.path.join(OUT_DIR, f"{complex_id}_pred.sdf")
    try:
        Chem.MolToMolFile(mol, out_path)
        success += 1
    except Exception as e:
        print(f"  [SKIP] {complex_id}: MolToMolFile failed: {e}")
        failed += 1

print(f"\n✅ Done: {success} SDF files written, {failed} failed → {OUT_DIR}")
