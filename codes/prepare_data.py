"""
prepare_data.py — turns our raw protein/ligand files into LMDB format for Uni-Mol.


Uni-Mol Docking V2 can't read .sdf/.pdb files directly. It needs everything
packed into an LMDB database in a specific dictionary format. This script
does that conversion for us.

For each protein-ligand pair, we:
  1. Read the ligand's real 3D shape from the .sdf file .
  2. Generate a few random 3D shapes for the same ligand with RDKit .
  3. Find which protein atoms are close to the ligand  —
     we use a simple 6 Å cutoff, same as the original paper
  4. Re-center everything so the pocket sits at (0,0,0)
  5. Save it all into one LMDB entry

Two LMDB files come out of this: train.lmdb  and
valid.lmdb .


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
    to Make sure a set of coordinates won't break training.

    The model's loss uses sqrt(distance) between atoms. If two atoms end up
    at the exact same spot, that distance is 0, and the gradient of sqrt(0)
    is infinity — this is what causes the whole training run to crash with
    NaN gradients, even though nothing looks wrong if you just print the loss.

    So before anything gets used, we check: no NaN/Inf values, and no two
    atoms closer than min_dist (default 0.5 Å, since real atoms never get
    that close).

    Returns (True, None) if it's fine. Otherwise returns (False, reason) where reason is a string explaining

    """
    if np.isnan(coords).any() or np.isinf(coords).any():
        return False, f"{name}: contains NaN/Inf"

    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 10.0)  # ignore distance-to-self

    min_d = dists.min()
    if min_d < min_dist:
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        return False, f"{name}: atoms {idx[0]}&{idx[1]} are {min_d:.4f} Å apart (< {min_dist} Å)"

    return True, None


# ============================================================
# Conformer generation
# ============================================================

# A conformer is just one possible 3D shape a flexible molecule can have,
#since molecules can rotate around their bonds. We generate a handful of
#these as the model's input — the model then learns to fix them into the
#correct binding pose.
 
def generate_conformers(mol, num_confs=20, seed=42):
    """
    Generate a few candidate 3D shapes for a ligand using RDKit.

    - Adds explicit hydrogens (the model needs to see them, RDKit hides them by default)
    - Generates `num_confs` random 3D shapes
    - Runs a quick MMFF optimization on each so they look chemically reasonable

    seed is fixed so the same molecule always gives the same shapes (it's reproducible).
    """
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
            pass  # some conformers just fail to optimize, skip and move on
    return mol


def filter_conformers(mol, max_keep=5):
    """Keep only the 'safe' conformers, stop once we have max_keep."""
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
# We don't feed the model the whole protein  — just the local
# area around where the ligand binds.

def extract_pocket(protein_path, ligand_coords_noH, cutoff=6.0):
    """
    Grab every protein atom within cutoff Å of the ligand.

    This is the same idea as the original paper — since we already have the
    real crystal structure, we know exactly where the ligand sits, so we
    don't need any fancy docking-grid setup, just a simple distance cutoff.

    Returns (patoms, pcoords) — element symbols + coordinates of pocket atoms,
    or None if something looks wrong with the result.
    """
    pmol = PandasPdb().read_pdb(protein_path)
    pdf = pmol.df['ATOM']

    atom_names = pdf['atom_name'].str.strip().tolist()
    coords = pdf[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)

    # distance from each protein atom to the closest ligand atom
    dists = np.linalg.norm(
        coords[:, None, :] - ligand_coords_noH[None, :, :], axis=-1
    ).min(axis=1)

    mask = dists <= cutoff

    # PDB files name atoms like "CA", "OD1", "NE2" — the model only knows
    # plain element symbols (C, N, O, S, H), so we convert here.
    VALID_ELEMENTS = {'C', 'N', 'O', 'S', 'H'}

    def pdb_name_to_element(name):
        name = name.strip()
        if not name:
            return 'C'
        if name[0].isdigit():
            # names like "1HB" -> the element letter is the second character
            elem = name[1] if len(name) > 1 else 'C'
        else:
            elem = name[0]
        return elem if elem in VALID_ELEMENTS else 'C'  # default to C if unknown

    patoms = [pdb_name_to_element(a) for a, m in zip(atom_names, mask) if m]
    pcoords = coords[mask]

    # if we got a suspiciously small pocket, widen the search and try again
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
# This is where one full protein-ligand complex gets turned into one LMDB entry.

def process_pair(ligand_path, protein_path, pdbid, num_confs=20, max_keep=5):
    """

    pdbid here is actually the full unique complex id (e.g. "3ix2_AC2_A_302"),
    not just the 4-letter PDB code — a single PDB code can have multiple
    different ligand copies in it, so we need the full id to keep them apart.

    """
    print(f"Processing {pdbid}", flush=True)

    # load the ligand's real crystal structure
    supp = Chem.SDMolSupplier(ligand_path, removeHs=False)
    mols = [m for m in supp if m is not None]
    if not mols:
        print(f"  → could not load ligand: {ligand_path}")
        return None

    mol = mols[0]
    # add hydrogens if they're not already there.
    mol_withH = Chem.AddHs(mol) if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() else mol
    holo_coords = mol_withH.GetConformer().GetPositions().astype(np.float32)
    atom_symbols = [a.GetSymbol() for a in mol_withH.GetAtoms()]
    smi = Chem.MolToSmiles(mol_withH)

    ok, reason = check_coords(holo_coords, "holo")
    if not ok:
        print(f"  → skipping {pdbid}: {reason}")
        return None

    # generate the model's input shapes
    rdkit_mol = generate_conformers(mol, num_confs=num_confs)
    conformers = filter_conformers(rdkit_mol, max_keep=max_keep)
    if len(conformers) == 0:
        print(f"  → no valid conformers for {pdbid}")
        return None

    # for pocket distance we use heavy atoms only 
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

    # center everything on the pocket — the pocket stays put while the
    # ligand is the thing that moves during docking, so it makes sense
    # to use the pocket as our fixed reference point at (0,0,0)
    center = pcoords.mean(axis=0)
    holo_coords = holo_coords - center
    pcoords     = pcoords - center
    # each conformer gets centered on its own mean — this matches how RDKit
    # generates them near the origin, so the model learns the move needed
    # to go from centered on itself to centered on the pocket.
    conformers  = [c - c.mean(axis=0) for c in conformers]

    entry = {
        "atoms":                   atom_symbols,
        "coordinates":             conformers,            # model input shapes
        "holo_coordinates":        [holo_coords],         
        "pocket_atoms":            [[a] for a in patoms],
        "pocket_coordinates":      [pcoords],
        "holo_pocket_coordinates": [pcoords],
        "pocket":                  pdbid,                 
        "smi":                     smi,                   # used to rebuild the molecule during inference
    }
    return entry


# ============================================================
# Build LMDB from CSV
# ============================================================

def make_lmdb(csv_path, data_dir, output_lmdb, num_confs=20, max_keep=5, limit=None):
    """
    Loop through every row in a CSV and build an LMDB file from it.

    limit lets you only process the first N rows .
    """
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)
        print(f"   DEBUG MODE: only processing first {limit} rows")

    # clear out any old LMDB so we don't mix old and new data
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

        # the CSV's PDBID column  isn't unique enough on its own,
        # since one PDB code can have several different ligand copies. So we
        # build a proper unique id from the ligand filename instead.
        # This is also the exact filename format evaluation.py expects later.

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

        # save progress every 100 entries instead of only at the very end

        if count % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  ... {count} samples written so far")

    txn.commit()
    env.close()
    print(f" Done: {count} samples saved, {failed} failed → {output_lmdb}")


# ============================================================
# RUN
# ============================================================

BASE = '/home/bdldt_team007/DLDockingBenchSeminar'

print("=== TRAIN ===")
make_lmdb(
    csv_path    = f'{BASE}/data/proto_train.csv',
    data_dir    = f'{BASE}/data/proto_train',
    output_lmdb = f'{BASE}/data/processed/train.lmdb',
    num_confs   = 20,
    max_keep    = 5,
    limit       = None,
)

print("=== VALID ===")
make_lmdb(
    csv_path    = f'{BASE}/data/proto_test.csv',
    data_dir    = f'{BASE}/data/proto_test',
    output_lmdb = f'{BASE}/data/processed/valid.lmdb',
    num_confs   = 20,
    max_keep    = 10,   
                        
    limit       = None,
)
