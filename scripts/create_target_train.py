import pandas as pd

# 1. Define file paths
input_file = (
    "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_train_processed.csv"
)
output_file = (
    "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_train_final.csv"
)

# 2. Load the dataset
df = pd.read_csv(input_file)

# 3. Extract the base identifier (e.g., "10gs_VWW_A_210")
# This splits by '_ligand_refined' and grabs everything before it
complex_ids = df["ligand_file_name"].str.split("_ligand_refined").str[0]

# 4. Insert it as the very first column (index 0) named 'complex_id'
df.insert(0, "Target", complex_ids)

# 5. Save the final file
df.to_csv(output_file, index=False)

print("Column inserted successfully! Preview of the new format:")
print(df[["Target", "ligand_file_name", "PDBID", "pIC50"]].head())