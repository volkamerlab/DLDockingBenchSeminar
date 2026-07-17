#!/bin/bash
# "." (current directory) is the project root.

PROJECT_ROOT="$(pwd)"

echo "Running in: $PROJECT_ROOT"
ls -F # Debug: List files to verify they were transferred

TRAIN_FOLDER=~/DLDockingBenchSeminar/data
DOCK_FOLDER=posebusters_filtered/

# export PYTHONPATH="$PROJECT_ROOT/interformer:$PYTHONPATH"
PYTHONPATH=interformer/ 

# Use relative paths from the project root
# Add the checkpoint ensemble
# To obtain the dock_poses to retrain the affinity&pose
# python interformer/inference.py \
    # -test_csv data/proto_train_final.csv \
    # Results obtained - complex, gaussian_predict, uff, ligand

python interformer/inference.py -test_csv posebusters_filtered.csv \
    -work_path posebusters_filtered \
    -ensemble checkpoints/energy_model \
    -ligand_folder ligand/rcsb \
    -gpus 1 \
    -batch_size 1 \
    -posfix *energy_model* \
    -energy_output_folder /home/bdldt_team001/DLDockingBenchSeminar/dock_results/posebusters \
    -reload

