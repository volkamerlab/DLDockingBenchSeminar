#!/bin/bash
# "." (current directory) is the project root.

PROJECT_ROOT="$(pwd)"

echo "Running in: $PROJECT_ROOT"
ls -F # Debug: List files to verify they were transferred

# export PYTHONPATH="$PROJECT_ROOT/interformer:$PYTHONPATH"
PYTHONPATH=interformer/ 

# Scoring the docking pose, this step will generate a tmp_beta folder. Ensure that you delete this cache before running a new prediction.
python interformer/inference.py -test_csv proto_test_final.round0.csv \
-work_path proto_test \
-ligand_folder /ligand \
-ensemble checkpoints/affinity_normal_model \
-gpus 1 \
-batch_size 10 \
-posfix *affinity_normal_model* \
--pose_sel True


