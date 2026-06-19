#!/bin/bash
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base

cd /home/bdldt_team001/DLDockingBenchSeminar

#preprocessing portion
# Affinity&PoseScore
# PYTHONPATH=interformer python interformer/pre.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020_round0_full.csv \
# -work_path /opt/home/revoli/data_worker/interformer/poses \
# -filter_type full \
# -dataset sbdd \
# -reload \
# -affinity_pre

# training portion
echo "Running from: $CURRENT_DIR" \
# PYTHONPATH=interformer/ \
export PYTHONPATH="/home/bdldt_team001/DLDockingBenchSeminar/interformer:$PYTHONPATH"
# Affinity and pose model
# Proper path to be mentioned
python train.py -data_path data/proto_train_final.round0.csv \
-work_path data/proto_train \
-seed 1111 \
-filter_type full \
-native_sampler 0 \
-Code affinity \
-batch_size 10 \
-gpus 1 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
--warmup_updates 10000 \
--peak_lr 0.0008 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5 \
--pose_sel_mode True