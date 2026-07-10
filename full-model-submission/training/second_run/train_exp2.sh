#!/bin/bash
BASE="/home/bdldt_team007/DLDockingBenchSeminar/second_run"
data_path="$BASE/data"
save_dir="$BASE/checkpoints"
finetune_mol_model="$BASE/weights/mol_pre_no_h_220816.pt"
finetune_pocket_model="$BASE/weights/pocket_pre_220816.pt"
lr=1e-5
batch_size=8
batch_size_valid=8
conf_size=10
epoch=15
dropout=0.2
update_freq=1
dist_threshold=8.0
recycling=3
mkdir -p $save_dir
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "=== EXP2 TRAIN COMMAND START ==="
$(which unicore-train) $data_path \
  --user-dir /home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/unimol \
  --train-subset train \
  --valid-subset valid \
  --num-workers 0 \
  --task docking_pose_v2 \
  --loss docking_pose_v2 \
  --arch docking_pose_v2 \
  --optimizer adam \
  --adam-betas '(0.9, 0.99)' \
  --adam-eps 1e-6 \
  --clip-norm 1.0 \
  --lr-scheduler polynomial_decay \
  --lr $lr \
  --warmup-updates 2637 \
  --total-num-update 43965 \
  --max-epoch $epoch \
  --batch-size $batch_size \
  --batch-size-valid $batch_size_valid \
  --conf-size $conf_size \
  --fp16 \
  --mol-pooler-dropout $dropout \
  --pocket-pooler-dropout $dropout \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --update-freq $update_freq \
  --seed 42 \
  --log-interval 50 \
  --log-format simple \
  --validate-interval 1 \
  --save-interval 1 \
  --keep-last-epochs 15 \
  --save-interval-updates 500 \
  --finetune-mol-model $finetune_mol_model \
  --finetune-pocket-model $finetune_pocket_model \
  --dist-threshold $dist_threshold \
  --recycling $recycling \
  --save-dir $save_dir
echo "=== EXP2 TRAIN COMMAND END ==="
