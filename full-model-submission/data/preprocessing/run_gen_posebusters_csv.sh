#!/bin/bash
# run_gen_posebusters_csv.sh — auto-generates posebusters_filtered.csv
#
# The LP-HiQBind Zenodo release includes posebusters_filtered.zip with SDF/PDB
# files but NO accompanying CSV file. Our evaluation.py needs a CSV with
# ligand_file, protein_file, and ligand_name columns to know which complexes
# to evaluate.
#
# This script generates that CSV automatically by reading the filenames
# directly from the posebusters_filtered.
#
# Input:  data/posebusters_filtered/posebusters_filtered/ (extracted zip)
# Output: data/posebusters_filtered.csv
#
# Run once before running run_prepare_posebusters.sh

python3 - << 'PYEOF'
import os
import pandas as pd

data_dir = "/home/bdldt_team007/DLDockingBenchSeminar/data/posebusters_filtered/posebusters_filtered"
files = os.listdir(data_dir)
ligands = sorted([f for f in files if f.endswith("_ligand.sdf")])

rows = []
for lig in ligands:
    prot = lig.replace("_ligand.sdf", "_protein.pdb")
    if os.path.exists(os.path.join(data_dir, prot)):
        rows.append({"ligand_file": lig, "protein_file": prot})

df = pd.DataFrame(rows)
df["ligand_name"] = df["ligand_file"].apply(lambda x: x.replace("_ligand.sdf", ""))
df.to_csv("/home/bdldt_team007/DLDockingBenchSeminar/data/posebusters_filtered.csv", index=False)
print(f"Done: {len(df)} complexes")
print(df.head(3).to_string())
PYEOF
