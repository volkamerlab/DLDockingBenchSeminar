"""
pkl_to_sdf.py — turns the model's raw predictions into actual .sdf pose files.

After we run infer.py, we get a file called valid.pkl. It has all the
predicted coordinates in it.

This script takes that valid.pkl and turns it into one .sdf file per
complex, which is the format evaluation.py actually wants.

Why we wrote our own version instead of using the one in the official repo:
The official Processor class (interface/predictor/processor.py) expects the
LMDB to have a "mol_list" key with pre-saved RDKit molecule objects in it.
Our LMDB doesn't have that — ours was built differently (see prepare_data.py)
and only has a "smi" (SMILES) field. So instead we just rebuild the molecule
from the SMILES string ourselves, which works just as well.

What this script actually does, step by step:
  1. Load the list of complex_ids from valid.lmdb, in order
  2. Load valid.pkl and pull out the predicted coordinates for every pose
  3. For each complex, we generated 10 candidate poses (conf_size=10)
     pick whichever one has the best (lowest) prmsd_score, since that's
     the model's own confidence score
  4. Rebuild the molecule from its SMILES string with RDKit
  5. Set the molecule's atom positions to the predicted coordinates
  6. Save it as a .sdf file
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

# ── Step 1: get the complex_ids in the same order they were stored

env = lmdb.open(LMDB_FILE, subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=256)
txn = env.begin()
raw_keys = list(txn.cursor().iternext(values=False))
raw_keys_sorted = sorted(raw_keys, key=lambda k: int(k.decode()))

complex_ids = []
smiles_list = []
for k in raw_keys_sorted:
    data = pickle.loads(txn.get(k))
    complex_ids.append(data["pocket"])   # this field actually holds the full complex_id
    smiles_list.append(data["smi"])

print(f"Loaded {len(complex_ids)} complex_ids from LMDB")
print(f"First 3: {complex_ids[:3]}")

# ── Step 2: pull out the predicted coordinates from valid.pkl
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

# ── Step 3: for each complex, pick the best pose and save it as .sdf 
success = 0
failed  = 0

for i in range(n_complexes):
    complex_id = complex_ids[i]
    smi        = smiles_list[i]

    # each complex has CONF_SIZE poses sitting next to each other in the list
    start = i * CONF_SIZE
    end   = start + CONF_SIZE

    coords_tta  = coord_predict_list[start:end]
    scores_tta  = prmsd_score_list[start:end]
    centers_tta = holo_center_list[start:end]

    # lower prmsd_score = model is more confident this pose is correct,
    # so we just pick the one with the lowest score
    best_idx    = int(np.argmin(scores_tta))
    best_coords = coords_tta[best_idx]
    best_center = centers_tta[best_idx]

    # rebuild the molecule from its SMILES instead 

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"  [SKIP] {complex_id}: could not parse SMILES")
        failed += 1
        continue

    # remove Hs since coord_predict is heavy-atom-only 
    mol = Chem.RemoveHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        result = AllChem.EmbedMolecule(mol, randomSeed=42, clearConfs=True)
    if result == -1:
        print(f"  [SKIP] {complex_id}: EmbedMolecule failed")
        failed += 1
        continue

    if best_coords.shape[0] != mol.GetNumAtoms():
        # this would mean the SMILES and the model's prediction disagree on
        # atom count, which shouldn't normally happen but we check anyway
        print(f"  [SKIP] {complex_id}: atom count mismatch "
              f"(coords={best_coords.shape[0]}, mol={mol.GetNumAtoms()})")
        failed += 1
        continue

    # set every atom to its predicted position, adding back the center offset 

    conf = mol.GetConformer(0)
    for j in range(best_coords.shape[0]):
        x = float(best_coords[j, 0] + best_center[0])
        y = float(best_coords[j, 1] + best_center[1])
        z = float(best_coords[j, 2] + best_center[2])
        conf.SetAtomPosition(j, Chem.rdGeometry.Point3D(x, y, z))

    out_path = os.path.join(OUT_DIR, f"{complex_id}_pred.sdf")
    try:
        Chem.MolToMolFile(mol, out_path)
        success += 1
    except Exception as e:
        print(f"  [SKIP] {complex_id}: MolToMolFile failed: {e}")
        failed += 1

print(f"\n Done: {success} SDF files written, {failed} failed → {OUT_DIR}")
