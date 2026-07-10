#!/bin/bash
export PYTHONNOUSERSITE=1
cd /home/bdldt_team007/DLDockingBenchSeminar

python3 - << 'PYEOF'
import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb
import sys
sys.path.insert(0, '/home/bdldt_team007/DLDockingBenchSeminar')
from prepare_data import generate_conformers, filter_conformers, check_coords

BASE = '/home/bdldt_team007/DLDockingBenchSeminar'
csv_path = f'{BASE}/data/posebusters_filtered.csv'
data_dir = f'{BASE}/data/posebusters_filtered/posebusters_filtered'
output_lmdb = f'{BASE}/data/processed_posebusters/posebusters_filtered.lmdb'
POCKET_RADIUS = 6.0
num_confs = 20
max_keep = 10

df = pd.read_csv(csv_path)
print(f"Processing {len(df)} complexes")

for path in [output_lmdb, output_lmdb + "-lock"]:
    if os.path.exists(path):
        os.remove(path)

os.makedirs(os.path.dirname(output_lmdb), exist_ok=True)
env = lmdb.open(output_lmdb, map_size=int(5e9), subdir=False)

count = 0
failed = 0

for _, row in df.iterrows():
    lig_file = row['ligand_file']
    prot_file = row['protein_file']
    complex_id = lig_file.replace('_ligand.sdf', '')
    lig_path = os.path.join(data_dir, lig_file)
    prot_path = os.path.join(data_dir, prot_file)

    if not os.path.exists(lig_path) or not os.path.exists(prot_path):
        print(f"  missing: {complex_id}")
        failed += 1
        continue

    try:
        # Load WITHOUT hydrogens for coordinate check
        supplier_no_h = Chem.SDMolSupplier(lig_path, removeHs=True, sanitize=True)
        mol_no_h = next((m for m in supplier_no_h if m is not None), None)
        if mol_no_h is None:
            failed += 1
            continue

        # Check coordinates on no-H version (avoids false failures)
        holo_coords_noH = mol_no_h.GetConformer().GetPositions().astype(np.float32)
        ok, reason = check_coords(holo_coords_noH, "holo")
        if not ok:
            print(f"  skipping {complex_id}: {reason}")
            failed += 1
            continue

        # Load WITH hydrogens for storage (matches training LMDB format)
        supplier_with_h = Chem.SDMolSupplier(lig_path, removeHs=False, sanitize=True)
        mol_with_h = next((m for m in supplier_with_h if m is not None), None)
        if mol_with_h is None:
            failed += 1
            continue

        # Add H explicitly to ensure consistency
        mol_with_h = Chem.AddHs(mol_with_h)
        holo_coords = mol_with_h.GetConformer().GetPositions().astype(np.float32)
        atoms = [mol_with_h.GetAtomWithIdx(i).GetSymbol() for i in range(mol_with_h.GetNumAtoms())]
        smi = Chem.MolToSmiles(mol_with_h, isomericSmiles=True)

        # generate_conformers internally calls AddHs on mol_no_h → same atom count as mol_with_h
        mol_confs = generate_conformers(mol_no_h, num_confs=num_confs)
        conformers = filter_conformers(mol_confs, max_keep=max_keep)
        if not conformers:
            failed += 1
            continue

        # Verify atom count matches
        if len(conformers[0]) != len(atoms):
            print(f"  skipping {complex_id}: atom count mismatch conf={len(conformers[0])} atoms={len(atoms)}")
            failed += 1
            continue

        # Pocket from protein
        ppdb = PandasPdb().read_pdb(prot_path)
        prot_df = ppdb.df['ATOM']
        prot_coords = prot_df[['x_coord','y_coord','z_coord']].values.astype(np.float32)
        prot_atoms = prot_df['atom_name'].tolist()

        lig_center = holo_coords_noH.mean(axis=0)
        dists = np.linalg.norm(prot_coords - lig_center, axis=1)
        mask = dists < POCKET_RADIUS
        if mask.sum() == 0:
            failed += 1
            continue

        pocket_coords = prot_coords[mask]
        pocket_atoms = [prot_atoms[i] for i in np.where(mask)[0]]

        ok, reason = check_coords(pocket_coords, "pocket")
        if not ok:
            failed += 1
            continue

        center = pocket_coords.mean(axis=0)
        holo_coords_centered = holo_coords - center
        pocket_coords_centered = pocket_coords - center
        conformers_centered = [c - c.mean(axis=0) for c in conformers]

        entry = {
            "atoms":                   atoms,
            "coordinates":             conformers_centered,
            "holo_coordinates":        [holo_coords_centered],
            "pocket_atoms":            [[a] for a in pocket_atoms],
            "pocket_coordinates":      [pocket_coords_centered],
            "holo_pocket_coordinates": [pocket_coords_centered],
            "holo_center_coordinates": center,
            "pocket":                  complex_id,
            "smi":                     smi,
        }

        with env.begin(write=True) as txn:
            txn.put(f"{count}".encode("ascii"), pickle.dumps(entry))
        count += 1

        if count % 20 == 0:
            print(f"  ... {count} saved")

    except Exception as e:
        print(f"  error {complex_id}: {e}")
        failed += 1

env.close()
print(f"Done: {count} saved, {failed} failed -> {output_lmdb}")
PYEOF
