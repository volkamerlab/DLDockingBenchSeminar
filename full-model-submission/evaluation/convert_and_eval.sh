#!/bin/bash
cd /home/bdldt_team007/DLDockingBenchSeminar

python3 - << 'PYEOF'
import pandas as pd
df = pd.read_csv("data/full_val.csv")
def build_id(row):
    return f"{row['PDBID']}_{row['Ligand Name']}_{row['Ligand Chain']}_{row['Ligand Residue Number']}"
df["ligand_file_name"] = df.apply(lambda r: build_id(r) + "_ligand_refined.sdf", axis=1)
df["protein_file_name"] = df.apply(lambda r: build_id(r) + "_protein_refined.pdb", axis=1)
df.to_csv("data/full_val_converted.csv", index=False)
print(f"Done: {len(df)} rows")
PYEOF

cp data/full_val_converted.csv data/full_val.csv

ln -sfn /home/bdldt_team007/DLDockingBenchSeminar/data/full_data/full_val \
        /home/bdldt_team007/DLDockingBenchSeminar/data/full_val

ln -sfn /home/bdldt_team007/DLDockingBenchSeminar/results/full_val \
        /home/bdldt_team007/DLDockingBenchSeminar/results/full_val

python3 evaluation.py \
    --dataset full_val \
    --no_pb_valid \
    --output_csv run6/logs/full_val_evaluation.csv
