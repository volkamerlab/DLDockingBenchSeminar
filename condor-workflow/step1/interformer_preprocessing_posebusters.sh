#!/bin/bash
### Start Interformer preprocessing code
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base
# Install missing dependencies to the container's user space
# pip install --user seaborn scikit-learn safetensors tqdm pandas transformers

# Add the local install path to Python's search path
# export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages

cd /home/bdldt_team001/DLDockingBenchSeminar

mkdir -p posebusters_filtered/ligand/rcsb
mkdir -p posebusters_filtered/uff
mkdir -p posebusters_filtered/pocket

# Preprocess
# Add H atoms to ligand molecules.
for f in posebusters_filtered/sdf/*.sdf; do 
    obabel "$f" -p 7.4 -O "posebusters_filtered/ligand/rcsb/$(basename "$f")" 
done


# Generate inital ligand conformation using UFF (or any other ligand prepare program of your choice).  
# ligand/rcsb/ is the reference ligand
# uff is the optimized conformation that we can experiment on 
python tools/rdkit_ETKDG_3d_gen.py posebusters_filtered/ligand/rcsb/ posebusters_filtered/uff/ 


####
# Protein
# Use the Reduce program to preprocess the entire protein.
# Adjust the protein structure to prevent steric clashes.
for pdb in posebusters_filtered/pdb/*.pdb; do 
    reduce -r "$pdb" > "posebusters_filtered/pocket/$(basename "$pdb")" 
done


# Extract the pocket within 10 Å around the reference ligand. The third argument 1 indicates removal of the CCD ligand from the PDB, use 0 if you do not wish to remove it.
python tools/extract_pocket_by_ligand.py posebusters_filtered/pocket/ posebusters_filtered/ligand/rcsb/ 0 && mv posebusters_filtered/pocket/output/*.pdb posebusters_filtered/pocket
## End of Interformer preprocessing code

# # Naming convention to match the OG data
python3 -u scripts/rename_docked_sdf_train.py 
