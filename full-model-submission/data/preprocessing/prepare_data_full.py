"""
prepare_data_full.py — preprocessing for the LP-HiQBind full dataset.

Same idea as prepare_data.py, but adapted for the new CSV format. The old
proto_train.csv had a ligand_file_name / protein_file_name column we could
read directly. This new full_train.csv / full_val.csv doesn't -- instead we
have to BUILD the filename ourselves from a few separate columns:

    PDBID + Ligand Name + Ligand Chain + Ligand Residue Number
    -> "{PDBID}_{LigandName}_{LigandChain}_{LigandResidueNumber}_ligand_refined.sdf"

Everything else (conformer generation, pocket extraction, the nan_to_num-style
safety checks) is identical to prepare_data.py -- copy logic unchanged below,
just the filename-building and CSV reading part is new.
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
# Coordinate validity check (unchanged from prepare_data.py)
# ============================================================

def check_coords(coords, name, min_dist=0.5):
    """Same as in prepare_data.py -- rejects NaN/Inf or atoms sitting on top of each other."""
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
# Conformer generation (unchanged from prepare_data.py)
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
# Pocket extraction (unchanged from prepare_data.py)
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

    VALID_ELEMENTS = {'C', 'N', 'O', 'S', 'H'}

    def pdb_name_to_element(name):
        name = name.strip()
        if not name:
            return 'C'
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
# Single pair processing (unchanged from prepare_data.py)
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

    ok, reason = check_coords(holo_coords, "holo")
    if not ok:
        print(f"  → skipping {pdbid}: {reason}")
        return None

    rdkit_mol = generate_conformers(mol, num_confs=num_confs)
    conformers = filter_conformers(rdkit_mol, max_keep=max_keep)
    if len(conformers) == 0:
        print(f"  → no valid conformers for {pdbid}")
        return None

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

    center = pcoords.mean(axis=0)
    holo_coords = holo_coords - center
    pcoords     = pcoords - center
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
# Build LMDB from the NEW CSV format
# ============================================================

def build_filename(row):
    """
    LP-HiQBind doesn't give us a ready-made filename column -- we build it
    ourselves from PDBID + Ligand Name + Ligand Chain + Ligand Residue Number.

    Example: PDBID=10gs, Ligand Name=VWW, Ligand Chain=B, Ligand Residue Number=210
             -> complex_id = "10gs_VWW_B_210"
             -> ligand file = "10gs_VWW_B_210_ligand_refined.sdf"
             -> protein file = "10gs_VWW_B_210_protein_refined.pdb"
    """
    complex_id = f"{row['PDBID']}_{row['Ligand Name']}_{row['Ligand Chain']}_{row['Ligand Residue Number']}"
    return complex_id


def make_lmdb(csv_path, data_dir, output_lmdb, num_confs=20, max_keep=5, limit=None):
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)
        print(f"  ⚠️  DEBUG MODE: only processing first {limit} rows")

    for path in [output_lmdb, output_lmdb + "-lock"]:
        if os.path.exists(path):
            os.remove(path)

    os.makedirs(os.path.dirname(output_lmdb), exist_ok=True)

    env = lmdb.open(output_lmdb, map_size=int(2e11), subdir=False)  # bumped map_size for the much bigger dataset
    txn = env.begin(write=True)

    count = 0
    failed = 0

    for _, row in df.iterrows():
        complex_id = build_filename(row)
        lig_path  = os.path.join(data_dir, f"{complex_id}_ligand_refined.sdf")
        prot_path = os.path.join(data_dir, f"{complex_id}_protein_refined.pdb")

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

        if count % 500 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  ... {count} samples written so far")

    txn.commit()
    env.close()
    print(f"✅ Done: {count} samples saved, {failed} failed → {output_lmdb}")


# ============================================================
# RUN
# ============================================================
# Start with a small `limit` to sanity-check everything works on this new
# CSV format before committing to the full ~23,000 / ~2,600 complex run.

BASE = '/home/bdldt_team007/DLDockingBenchSeminar'

print("=== FULL TRAIN (TEST RUN, limit=20) ===")
make_lmdb(
    csv_path    = f'{BASE}/data/full_data/full_train.csv',
    data_dir    = f'{BASE}/data/full_data/full_train',
    output_lmdb = f'{BASE}/data/processed/full_train.lmdb',
    num_confs   = 20,
    max_keep    = 5,
    limit       = None,   # small test first -- raise/remove once confirmed working
)

print("=== FULL VAL (TEST RUN, limit=20) ===")
make_lmdb(
    csv_path    = f'{BASE}/data/full_data/full_val.csv',
    data_dir    = f'{BASE}/data/full_data/full_val',
    output_lmdb = f'{BASE}/data/processed/full_val.lmdb',
    num_confs   = 20,
    max_keep    = 10,
    limit       = None,
)
