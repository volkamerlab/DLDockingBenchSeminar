#!/bin/bash
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base
# Install missing dependencies to the container's user space
# pip install --user seaborn scikit-learn safetensors tqdm pandas transformers

# Add the local install path to Python's search path
# export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages

cd /home/bdldt_team001/DLDockingBenchSeminar

rm -r data/proto_train/ligand
rm -r data/proto_train/uff
rm -r data/proto_train/pocket

mkdir -p data/proto_train/ligand
mkdir -p data/proto_train/uff
mkdir -p data/proto_train/pocket

# Preprocess
#original code:
# obabel data/proto_train/train_sdf -p 7.4 -O data/ligand/
for f in data/proto_train/train_sdf/*.sdf; do 
    obabel "$f" -p 7.4 -O "data/proto_train/ligand/$(basename "$f")" 
done

# Generate inital ligand conformation using UFF (or any other ligand prepare program of your choice).  
python tools/rdkit_ETKDG_3d_gen.py data/proto_train/ligand/ data/proto_train/uff/ 
####
# Protein
# Use the Reduce program to preprocess the entire protein.
#original code:
# mkdir -p data/proto_train/pocket && reduce -r data/proto_train/train_pdb/ > data/proto_train/pocket
for pdb in data/proto_train/train_pdb/*.pdb; do 
    reduce -r "$pdb" > "data/proto_train/pocket/$(basename "$pdb")" 
done

# Extract the pocket within 10 Å around the reference ligand. The third argument 1 indicates removal of the CCD ligand from the PDB, use 0 if you do not wish to remove it.
python tools/extract_pocket_by_ligand.py data/proto_train/pocket/ data/proto_train/ligand/ 0 && mv data/proto_train/pocket/output/* data/proto_train/pocket

# for later (energy model)
# python3 -u train.py 
# cd /data
# unzip prototype_model_data.zip
# -data_path /proto_train.csv \
# -work_path /proto_train \
# -ligand ligand/rcsb \
# -seed 1111 \
# -filter_type normal \
# -native_sampler 0 \
# -Code Energy \
# -batch_size 24 \
# -gpus 4 \
# -method Gnina2 \
# -patience 30 \
# -early_stop_metric val_loss \
# -early_stop_mode min \
# -affinity_pre \
# --warmup_updates 11000 \
# --peak_lr 0.0012 \
# --n_layers 6 \
# --hidden_dim 128 \
# --num_heads 8 \
# --dropout_rate 0.1 \
# --attention_dropout_rate 0.1 \
# --weight_decay 1e-5 \
# --energy_mode True