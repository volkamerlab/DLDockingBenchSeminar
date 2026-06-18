'''
obabel produced suffixes with '_ligand_refined.sdf', but our scripts required that the suffix 
should have '_docked.sdf'.

Example scripts that use the '_docked.sdf' format:
test_pipline.py
obabel_api.py
bindingdata.py
+ more...
'''
import os

# Define the directory containing your SDF files
directory_path = "/home/bdldt_team001/DLDockingBenchSeminar/data/proto_test/ligand/rcsb"

print(f"Renaming files in: {directory_path}")
count = 0

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file matches your specific suffix
    if filename.endswith("_ligand_refined.sdf"):
        # Construct the old full path
        old_file = os.path.join(directory_path, filename)

        # Generate the new filename by replacing the suffix
        new_filename = filename.replace("_ligand_refined.sdf", "_docked.sdf")
        new_file = os.path.join(directory_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        count += 1

print(f"Successfully renamed {count} files to '_docked.sdf'!")