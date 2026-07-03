#!/bin/bash
# IMPORTANT: COMMENT OUT LINE 35

### Start Interformer preprocessing code
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base
# Install missing dependencies to the container's user space
# pip install --user seaborn scikit-learn safetensors tqdm pandas transformers

# Add the local install path to Python's search path
# export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages

cd /home/bdldt_team001/DLDockingBenchSeminar


# Validation set
# mkdir -p data/proto_val/ligand
# mkdir -p data/proto_val/ligand/rcsb
# mkdir -p data/proto_val/uff
# mkdir -p data/proto_val/pocket

# # Preprocess
# # Add H atoms to ligand molecules.
# for f in data/proto_val/val_sdf/*.sdf; do 
#     obabel "$f" -p 7.4 -O "data/proto_val/ligand/rcsb/$(basename "$f")" 
# done

# # Generate inital ligand conformation using UFF (or any other ligand prepare program of your choice).  
# # ligand/rcsb/ is the reference ligand
# # uff is the optimized conformation that we can experiment on 
# python tools/rdkit_ETKDG_3d_gen.py data/proto_val/ligand/rcsb/ data/proto_val/uff/ 

# ####
# # Protein
# # Use the Reduce program to preprocess the entire protein.
# # Adjust the protein structure to prevent steric clashes.
# for pdb in data/proto_val/val_pdb/*.pdb; do 
#     reduce -r "$pdb" > "data/proto_val/pocket/$(basename "$pdb")" 
# done

# Extract the pocket within 10 Å around the reference ligand. The third argument 1 indicates removal of the CCD ligand from the PDB, use 0 if you do not wish to remove it.
# python tools/extract_pocket_by_ligand.py data/proto_val/pocket/ data/proto_val/ligand/rcsb/ 0 && mv data/proto_val/pocket/output/*.pdb data/proto_val/pocket
### End of Interformer preprocessing code

### Lak&Ben Trainset Preprocessing ###
# # first manipulation to get uniprot and pIC50 values - (Skip - Hamza's response about how we handle the -ve pIC50 values - 25.6)
# python3 -u scripts/get_IC50_uniprot_train.py

# # split the ligand_file_name into target separately (Changes made to train.py file according to naming conventions provided in the full dataset)
# python3 -u scripts/create_target_train.py 

# # Naming convention to match the OG data
# python3 -u scripts/rename_docked_sdf_train.py 

# Final combine csv's
python3 -u scripts/combine_csv.py
