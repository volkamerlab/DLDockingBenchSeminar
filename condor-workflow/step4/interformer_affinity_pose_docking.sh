#!/bin/bash
# "." (current directory) is the project root.
set -e  # <-- Exit immediately if any command fails

PROJECT_ROOT="$(pwd)"

echo "Running in: $PROJECT_ROOT"
ls -F # Debug: List files to verify they were transferred

TRAIN_FOLDER=~/DLDockingBenchSeminar/data
DOCK_FOLDER=dock_results/energy_train

# mkdir -p "$DOCK_FOLDER"
# mkdir -p "$DOCK_FOLDER"/ligand_reconstructing

# for tarball in complex gaussian_predict ligand uff; do
#     if [ -f "${tarball}.tar.gz" ]; then
#         echo "=== Unpacking ${tarball}.tar.gz ==="
#         tar -xzf "${tarball}.tar.gz" -C "$DOCK_FOLDER" #unzip
#         rm -f "${tarball}.tar.gz"   # <-- free the compressed copy immediately
#     fi
# done
# ls -F "$DOCK_FOLDER" 

PYTHONPATH=interformer/ 
# echo "PYTHONPATH=$PYTHONPATH"
# export PYTHONPATH="$PROJECT_ROOT/interformer:$PYTHONPATH"

# CSV file for training the affinity and pose model
# Run each command seperately
# Start docking

echo "=== Installing PyVina Dependencies ==="
pip install --user --no-cache-dir ./docking

# Ensure that the logs file is available 
mkdir -p logs

# If you planning to use uff ligand conformation to dock, you can use argument `--uff_folder uff`
export OMP_NUM_THREADS="32,32"
python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER --find_all find

# Next run - comment out line 20-23 and run line 27.
# Make a docking summary csv 
python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all stat

# Merging original csv with the docking summary, gather information of rmsd, enery, num_torsions and poserank(cid, id of the conformation in a sdf)
# python docking/merge_summary_input.py $DOCK_FOLDER/ligand_reconstructing/stat_concated.csv proto_train_val_final.csv

# python label_negatives.py proto_train_val_final.round0.csv proto_train_val_final_w_neg_labels.round0.csv --guarantee-positive

echo "=== Docking step complete ==="
# Validate if the file path exists before running find to avoid ugly errors
if [ -d "${DOCK_FOLDER}/ligand_reconstructing" ]; then
    find "${DOCK_FOLDER}/ligand_reconstructing" -name "*.sdf" | wc -l
else
    echo "ERROR: ${DOCK_FOLDER}/ligand_reconstructing still does not exist!"
    exit 1
fi