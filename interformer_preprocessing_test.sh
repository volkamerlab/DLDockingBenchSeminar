#!/bin/bash
### Start Interformer preprocessing code
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base
# Install missing dependencies to the container's user space
# pip install --user seaborn scikit-learn safetensors tqdm pandas transformers

# Add the local install path to Python's search path
# export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages

cd /home/bdldt_team001/DLDockingBenchSeminar

rm -r data/proto_test/ligand
rm -r data/proto_test/ligand/rcsb
rm -r data/proto_test/uff
rm -r data/proto_test/pocket

mkdir -p data/proto_test/ligand
mkdir -p data/proto_test/ligand/rcsb
mkdir -p data/proto_test/uff
mkdir -p data/proto_test/pocket

# Preprocess
# Add H atoms to ligand molecules and protonation states are determined via obabel
for f in data/proto_test/test_sdf/*.sdf; do 
    obabel "$f" -p 7.4 -O "data/proto_test/ligand/rcsb/$(basename "$f")" 
done

# Generate initial ligand conformation using UFF (or any other ligand prepare program of your choice).  
python tools/rdkit_ETKDG_3d_gen.py data/proto_test/ligand/rcsb/ data/proto_test/uff/ 
####
# Protein
# Use the Reduce program to preprocess the entire protein.
# Adjust the protein structure to prevent steric clashes.
for pdb in data/proto_test/test_pdb/*.pdb; do 
    reduce -r "$pdb" > "data/proto_test/pocket/$(basename "$pdb")" 
done

# Extract the pocket within 10 Å around the reference ligand. The third argument 1 indicates removal of the CCD ligand from the PDB, use 0 if you do not wish to remove it.
python tools/extract_pocket_by_ligand.py data/proto_test/pocket/ data/proto_test/ligand/rcsb/ 0 && mv data/proto_test/pocket/output/*.pdb data/proto_test/pocket/


# first manipulation to get uniprot and pIC50 values
python3 -u scripts/get_IC50_uniprot_test.py
# Naming convention to match the OG data
python3 -u scripts/rename_docked_sdf_test.py 
# split the ligand_file_name into target separately
python3 -u scripts/create_target_test.py 

