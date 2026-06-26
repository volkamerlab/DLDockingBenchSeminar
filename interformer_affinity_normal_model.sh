#!/bin/bash
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base

cd /home/bdldt_team001/DLDockingBenchSeminar

# preprocessing portion
# Affinity Normal
#PYTHONPATH=interformer python interformer/pre.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
#-work_path /opt/home/revoli/data_worker/interformer/poses \
#-ligand ligand/rcsb \
#-filter_type normal \
#-dataset sbdd \
#-reload \
#-affinity_pre

# training portion
echo "Running from: $CURRENT_DIR" \
# PYTHONPATH=interformer/ \
export PYTHONPATH="/home/bdldt_team001/DLDockingBenchSeminar/interformer:$PYTHONPATH"
# Affinity normal model
python train.py -data_path data/proto_train_final.csv \
-work_path data/proto_train \
-ligand ligand/rcsb \
-seed 1111 \
-filter_type normal \
-native_sampler 1 \
-Code affinity_normal \
-batch_size 4 \
-gpus 1 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
-run_name Affinity_normal_model \
--warmup_updates 10000 \
--peak_lr 0.0008 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5
