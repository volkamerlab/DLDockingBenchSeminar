import sys
sys.path.insert(0, '/home/bdldt_team007/DLDockingBenchSeminar')
from prepare_data import make_lmdb

BASE = '/home/bdldt_team007/DLDockingBenchSeminar'

import pandas as pd
df = pd.read_csv(f'{BASE}/data/full_test/full_test.csv')
def build_id(row):
    return f"{row['PDBID']}_{row['Ligand Name']}_{row['Ligand Chain']}_{row['Ligand Residue Number']}"
df["ligand_file_name"] = df.apply(lambda r: build_id(r) + "_ligand_refined.sdf", axis=1)
df["protein_file_name"] = df.apply(lambda r: build_id(r) + "_protein_refined.pdb", axis=1)
df.to_csv(f'{BASE}/data/full_test/full_test_converted.csv', index=False)
print(f"Converted {len(df)} rows")

make_lmdb(
    csv_path    = f'{BASE}/data/full_test/full_test_converted.csv',
    data_dir    = f'{BASE}/data/full_test/full_test',
    output_lmdb = f'{BASE}/data/processed_test/test.lmdb',
    num_confs   = 20,
    max_keep    = 10,
    limit       = None,
)
