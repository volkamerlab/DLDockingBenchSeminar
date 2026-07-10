#!/bin/bash
cd /home/bdldt_team007/DLDockingBenchSeminar
python3 - << 'PYEOF'
import pandas as pd
df = pd.read_csv("data/full_test/full_test.csv")
def build_id(row):
    return f"{row['PDBID']}_{row['Ligand Name']}_{row['Ligand Chain']}_{row['Ligand Residue Number']}"
df["ligand_file_name"] = df.apply(lambda r: build_id(r) + "_ligand_refined.sdf", axis=1)
df["protein_file_name"] = df.apply(lambda r: build_id(r) + "_protein_refined.pdb", axis=1)
df.to_csv("data/full_test_data.csv", index=False)
print(f"Converted {len(df)} rows")
PYEOF
python3 evaluation.py --dataset full_test_data --no_pb_valid \
    --output_csv run6/logs/full_test_evaluation.csv
