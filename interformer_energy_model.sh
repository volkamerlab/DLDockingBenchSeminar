#!/bin/bash
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base

cd /home/bdldt_team001/DLDockingBenchSeminar

#preprocessing portion
# # Energy
# PYTHONPATH=interformer python interformer/pre.py -data_path /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train.csv \
# -work_path /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train \
# -filter_type normal \
# -dataset sbdd \
# -ligand_folder /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train/ligand/rcsb \
# -reload

# training portion
echo "Running from: $CURRENT_DIR" \
# PYTHONPATH=interformer/ \
export PYTHONPATH="/home/bdldt_team001/DLDockingBenchSeminar/interformer:$PYTHONPATH"
# energy model
python3 -u train.py \
-data_path data/proto_train_final.csv \
-work_path data/proto_train \
-ligand ligand/rcsb \
-seed 1111 \
-filter_type normal \
-native_sampler 0 \
-Code Energy \
-batch_size 20 \
-gpus 1 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
--warmup_updates 11000 \
--peak_lr 0.0012 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5 \
--energy_mode True