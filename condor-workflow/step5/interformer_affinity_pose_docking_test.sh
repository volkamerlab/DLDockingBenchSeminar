#!/bin/bash
# "." (current directory) is the project root.

PROJECT_ROOT="$(pwd)"

echo "Running in: $PROJECT_ROOT"
ls -F # Debug: List files to verify they were transferred

TRAIN_FOLDER=~/DLDockingBenchSeminar/data
DOCK_FOLDER=dock_results_test/energy_test

# export PYTHONPATH="$PROJECT_ROOT/interformer:$PYTHONPATH"
PYTHONPATH=interformer/ 

# CSV file for training the affinity and pose model
# Run each command seperately
# Start docking

echo "=== Installing PyVina Dependencies ==="
pip install --user --no-cache-dir ./docking
# # If you planning to use uff ligand conformation to dock, you can use argument `--uff_folder uff`
export OMP_NUM_THREADS="32,32"
python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER -y --find_all find

# Next run - comment out line 20-23 and run line 27.
# Make a docking summary csv 
python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all stat

# Merging original csv with the docking summary, gather information of rmsd, enery, num_torsions and poserank(cid, id of the conformation in a sdf)
# python docking/merge_summary_input.py $DOCK_FOLDER/ligand_reconstructing/stat_concated.csv proto_test_final.csv


