"""
prepare_data.py — Preprocessing for Uni-Mol Docking V2
"""

import os
import pickle
import lmdb
import copy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb


# ============================================================
# Coordinate validity check
# ============================================================

def check_coords(coords, name, min_dist=0.5):
    """
    Check a coordinate array for issues that cause NaN gradients.

    Uni-Mol's loss computes pairwise Euclidean distances via sqrt().
    The gradient of sqrt(x) is 1/(2*sqrt(x)) → infinite at x=0.
    Any two atoms at ~0 separation produce inf/NaN on the backward pass
    even though the forward loss value looks finite.

    Returns (True, None) if valid, (False, reason_string) if not.
    """
    if np.isnan(coords).any() or np.isinf(coords).any():
        return False, f"{name}: contains NaN/Inf"

    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 10.0)

    min_d = dists.min()
    if min_d < min_dist:
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        return False, f"{name}: atoms {idx[0]}&{idx[1]} are {min_d:.4f} Å apart (< {min_dist} Å)"

    return True, None


# ============================================================
# Conformer generation
# ============================================================

def generate_conformers(mol, num_confs=20, seed=42):
    mol = copy.deepcopy(mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed,
        clearConfs=True, enforceChirality=True,
    )
    for cid in range(mol.GetNumConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        except Exception:
            pass
    return mol


def filter_conformers(mol, max_keep=5):
    """Keep valid conformers — no NaN/Inf, no atom pair closer than 0.5 Å."""
    valid = []
    for conf in mol.GetConformers():
        coords = conf.GetPositions().astype(np.float32)
        ok, reason = check_coords(coords, "conformer")
        if not ok:
            continue
        valid.append(coords)
        if len(valid) == max_keep:
            break
    return valid


# ============================================================
# Pocket extraction
# ============================================================

def extract_pocket(protein_path, ligand_coords_noH, cutoff=6.0):
    pmol = PandasPdb().read_pdb(protein_path)
    pdf = pmol.df['ATOM']

    atom_names = pdf['atom_name'].str.strip().tolist()
    coords = pdf[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)

    dists = np.linalg.norm(
        coords[:, None, :] - ligand_coords_noH[None, :, :], axis=-1
    ).min(axis=1)

    mask = dists <= cutoff

    # Convert PDB atom names (e.g. "CA", "OD1", "NE2", "SD") to element symbols
    # (e.g. "C", "O", "N", "S"). The pocket dictionary only knows C, N, O, S, H.
    # Anything unmapped defaults to "C" which is in the dictionary.
    VALID_ELEMENTS = {'C', 'N', 'O', 'S', 'H'}

    def pdb_name_to_element(name):
        name = name.strip()
        if not name:
            return 'C'
        # PDB names starting with digit: element is second char (e.g. "1HB" -> "H")
        if name[0].isdigit():
            elem = name[1] if len(name) > 1 else 'C'
        else:
            elem = name[0]
        return elem if elem in VALID_ELEMENTS else 'C'

    patoms = [pdb_name_to_element(a) for a, m in zip(atom_names, mask) if m]
    pcoords = coords[mask]

    if len(patoms) < 5:
        print(f"  → pocket too small at {cutoff:.1f} Å, expanding to {cutoff * 2:.1f} Å")
        return extract_pocket(protein_path, ligand_coords_noH, cutoff * 2)

    ok, reason = check_coords(pcoords, "pocket")
    if not ok:
        print(f"  → invalid pocket: {reason}")
        return None

    return patoms, pcoords


# ============================================================
# Single pair processing
# ============================================================

def process_pair(ligand_path, protein_path, pdbid, num_confs=20, max_keep=5):
    print(f"Processing {pdbid}", flush=True)

    supp = Chem.SDMolSupplier(ligand_path, removeHs=False)
    mols = [m for m in supp if m is not None]
    if not mols:
        print(f"  → could not load ligand: {ligand_path}")
        return None

    mol = mols[0]
    mol_withH = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    holo_coords = mol_withH.GetConformer().GetPositions().astype(np.float32)
    atom_symbols = [a.GetSymbol() for a in mol_withH.GetAtoms()]
    smi = Chem.MolToSmiles(mol_withH)

    # ✅ Check holo (crystal) coordinates — this was missing before
    ok, reason = check_coords(holo_coords, "holo")
    if not ok:
        print(f"  → skipping {pdbid}: {reason}")
        return None

    # Generate and filter random conformers
    rdkit_mol = generate_conformers(mol, num_confs=num_confs)
    conformers = filter_conformers(rdkit_mol, max_keep=max_keep)
    if len(conformers) == 0:
        print(f"  → no valid conformers for {pdbid}")
        return None

    # Extract pocket
    supp_noH = Chem.SDMolSupplier(ligand_path, removeHs=True)
    mols_noH = [m for m in supp_noH if m is not None]
    ligand_coords_noH = (
        mols_noH[0].GetConformer().GetPositions().astype(np.float32)
        if mols_noH else holo_coords
    )

    pocket = extract_pocket(protein_path, ligand_coords_noH)
    if pocket is None:
        return None
    patoms, pcoords = pocket

    # Center everything on the pocket center.
    # The pocket center is the stable reference frame for the binding site.
    # ReAlignLigandDataset in docking_pose_v2.py uses the original uncentered
    # dataset, so we must provide coordinates already in a consistent frame.
    center = pcoords.mean(axis=0)
    # Center holo and pocket on pocket center
    holo_coords = holo_coords - center
    pcoords     = pcoords - center
    # Center each conformer on its own mean (RDKit conformers are near origin,
    # pocket is now at origin, so conformers should also be near origin)
    conformers  = [c - c.mean(axis=0) for c in conformers]

    entry = {
        "atoms":                   atom_symbols,
        "coordinates":             conformers,
        "holo_coordinates":        [holo_coords],
        "pocket_atoms":            [[a] for a in patoms],
        "pocket_coordinates":      [pcoords],
        "holo_pocket_coordinates": [pcoords],
        "pocket":                  pdbid,
        "smi":                     smi,
    }
    return entry


# ============================================================
# Build LMDB from CSV
# ============================================================

def make_lmdb(csv_path, data_dir, output_lmdb, num_confs=20, max_keep=5, limit=None):
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)
        print(f"  ⚠️  DEBUG MODE: only processing first {limit} rows")

    for path in [output_lmdb, output_lmdb + "-lock"]:
        if os.path.exists(path):
            os.remove(path)

    os.makedirs(os.path.dirname(output_lmdb), exist_ok=True)

    env = lmdb.open(output_lmdb, map_size=int(1e10), subdir=False)
    txn = env.begin(write=True)

    count = 0
    failed = 0

    for _, row in df.iterrows():
        lig_path  = os.path.join(data_dir, row['ligand_file_name'])
        prot_path = os.path.join(data_dir, row['protein_file_name'])
        pdbid     = row['PDBID']
        complex_id = row['ligand_file_name'].replace('_ligand_refined.sdf', '')

        if not os.path.exists(lig_path):
            print(f"  → missing ligand file: {lig_path}")
            failed += 1
            continue
        if not os.path.exists(prot_path):
            print(f"  → missing protein file: {prot_path}")
            failed += 1
            continue

        entry = process_pair(lig_path, prot_path, complex_id, num_confs=num_confs, max_keep=max_keep)

        if entry is None:
            failed += 1
            continue

        txn.put(f"{count}".encode("ascii"), pickle.dumps(entry))
        count += 1

        if count % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  ... {count} samples written so far")

    txn.commit()
    env.close()
    print(f"✅ Done: {count} samples saved, {failed} failed → {output_lmdb}")


# ============================================================
# RUN
# ============================================================

BASE = '/home/bdldt_team007/DLDockingBenchSeminar'

#print("=== TRAIN ===")
#make_lmdb(
#    csv_path    = f'{BASE}/data/proto_train.csv',
#    data_dir    = f'{BASE}/data/proto_train',
#    output_lmdb = f'{BASE}/data/processed/train.lmdb',
#    num_confs   = 20,
#    max_keep    = 5,
#    limit       = None,
#)

print("=== VALID ===")
make_lmdb(
    csv_path    = f'{BASE}/data/proto_test.csv',
    data_dir    = f'{BASE}/data/proto_test',
    output_lmdb = f'{BASE}/data/processed/valid.lmdb',
    num_confs   = 20,
    max_keep    = 10,
    limit       = None,
)
