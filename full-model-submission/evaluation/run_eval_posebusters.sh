#!/bin/bash
cd /home/bdldt_team007/DLDockingBenchSeminar

# Add ligand_name column to CSV
python3 - << 'PYEOF'
import pandas as pd
df = pd.read_csv("data/posebusters_filtered.csv")
if "ligand_name" not in df.columns:
    df["ligand_name"] = df["ligand_file"].apply(lambda x: x.replace("_ligand.sdf", ""))
    df.to_csv("data/posebusters_filtered.csv", index=False)
    print(f"Added ligand_name column, {len(df)} rows")
else:
    print("ligand_name already present")
PYEOF

# Install posebusters
pip install posebusters --quiet

# Run evaluation
python3 evaluation.py --dataset posebusters_filtered \
    --output_csv run6/logs/posebusters_evaluation.csv
