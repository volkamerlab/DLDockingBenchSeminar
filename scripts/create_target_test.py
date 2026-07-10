import pandas as pd

# 1. Define file paths
input_file = (
    "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_test_processed.csv"
)
output_file = (
    "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_test_final.csv"
)

# 2. Load the dataset
df = pd.read_csv(input_file)

# 3. Extract the base identifier (e.g., "10gs_VWW_A_210")
# Combines 4 different columns in the csv to obtain the target_name.
complex_ids = df["PDBID"] + "_" + df["Ligand Name"] + "_" + df["Ligand Chain"] + "_" + df["Ligand Residue Number"]

# 4. Insert it as the very first column (index 0) named 'complex_id'
df.insert(0, "Target", complex_ids)

# 5. Save the final file
df.to_csv(output_file, index=False)

print("Column inserted successfully! Preview of the new format:")
print(df[["Target", "PDBID", "pIC50"]].head())