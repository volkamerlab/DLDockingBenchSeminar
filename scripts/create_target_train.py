import pandas as pd

'''
This script is responsible for renaming conventions so that it fits our workflow nicer.
Depending on which dataset we want to produce, we comment out/in the respective target files.
'''

# 1. Define file paths
# input_file = (
#     "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_train_processed.csv"
# )
# output_file = (
#     "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_train_final.csv"
# )

# for validation set
input_file = (
    "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_val_processed.csv"
)
output_file = (
    "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_val_final.csv"
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