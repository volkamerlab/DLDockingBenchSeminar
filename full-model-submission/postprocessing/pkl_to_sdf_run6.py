import os
import pickle
import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = "/home/bdldt_team007/DLDockingBenchSeminar/run6"
PKL_FILE  = f"{BASE}/inference_results/valid.pkl"
LMDB_FILE = f"{BASE}/data/valid.lmdb"
OUT_DIR   = f"{BASE}/inference_results/sdf"
CONF_SIZE = 10

os.makedirs(OUT_DIR, exist_ok=True)

env = lmdb.open(LMDB_FILE, subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=256)
txn = env.begin()
raw_keys_sorted = sorted(txn.cursor().iternext(values=False), key=lambda k: int(k.decode()))

complex_ids = []
smiles_list = []
for k in raw_keys_sorted:
    data = pickle.loads(txn.get(k))
    complex_ids.append(data["pocket"])
    smiles_list.append(data["smi"])

print(f"Loaded {len(complex_ids)} complex_ids from LMDB")

with open(PKL_FILE, "rb") as f:
    batches = pickle.load(f)

print(f"Loaded {len(batches)} batches from pkl")

coord_predict_list, holo_center_list, prmsd_score_list = [], [], []

for batch in batches:
    for i in range(batch["atoms"].size(0)):
        token_mask = batch["atoms"][i] > 2
        coord_predict_list.append(batch["coord_predict"][i][token_mask].numpy().astype(np.float32))
        holo_center_list.append(batch["holo_center_coordinates"][i][:3].numpy().astype(np.float32))
        prmsd_score_list.append(batch["prmsd_score"][i].item())

n_complexes = len(coord_predict_list) // CONF_SIZE
print(f"Total poses: {len(coord_predict_list)}, complexes: {n_complexes}")
assert n_complexes == len(complex_ids), f"Mismatch: {n_complexes} vs {len(complex_ids)}"

success, failed = 0, 0

for i in range(n_complexes):
    complex_id = complex_ids[i]
    smi = smiles_list[i]
    start = i * CONF_SIZE
    scores_tta = prmsd_score_list[start:start+CONF_SIZE]
    best_idx = int(np.argmin(scores_tta))
    best_coords = coord_predict_list[start+best_idx]
    best_center = holo_center_list[start+best_idx]

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"  [SKIP] {complex_id}: could not parse SMILES")
        failed += 1
        continue

    mol = Chem.RemoveHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        if AllChem.EmbedMolecule(mol, randomSeed=42, clearConfs=True) == -1:
            print(f"  [SKIP] {complex_id}: EmbedMolecule failed")
            failed += 1
            continue

    if best_coords.shape[0] != mol.GetNumAtoms():
        print(f"  [SKIP] {complex_id}: atom count mismatch (coords={best_coords.shape[0]}, mol={mol.GetNumAtoms()})")
        failed += 1
        continue

    conf = mol.GetConformer(0)
    for j in range(best_coords.shape[0]):
        conf.SetAtomPosition(j, Chem.rdGeometry.Point3D(
            float(best_coords[j,0]+best_center[0]),
            float(best_coords[j,1]+best_center[1]),
            float(best_coords[j,2]+best_center[2])))

    try:
        Chem.MolToMolFile(mol, os.path.join(OUT_DIR, f"{complex_id}_pred.sdf"))
        success += 1
    except Exception as e:
        print(f"  [SKIP] {complex_id}: {e}")
        failed += 1

print(f"\nDone: {success} SDF files written, {failed} failed -> {OUT_DIR}")
