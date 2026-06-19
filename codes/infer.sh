#!/bin/bash
# infer.sh — runs our trained model on the test set to generate predicted poses.
# Called by run_infer.sh, which is called by condor_infer.sub.
# Output: results/proto_test/valid.pkl 

BASE="/home/bdldt_team007/DLDockingBenchSeminar"
data_path="$BASE/data/processed"                                  # where valid.lmdb lives
results_path="$BASE/results/proto_test"                           # valid.pkl gets written here
checkpoint="$BASE/results/checkpoints/checkpoint_best.pt"          # our fine-tuned model

mkdir -p $results_path

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

echo "=== INFERENCE START ==="

# infer.py is the official Uni-Mol inference script. Note --recycling 1
# here matches what we used during training — using a different recycling
# value than training would be inconsistent with how the model learned.
python /home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/unimol/infer.py \
  --user-dir /home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/unimol \
  $data_path \
  --valid-subset valid \
  --results-path $results_path \
  --num-workers 0 \
  --ddp-backend=c10d \
  --batch-size 4 \
  --task docking_pose_v2 \
  --loss docking_pose_v2 \
  --arch docking_pose_v2 \
  --conf-size 10 \
  --dist-threshold 8.0 \
  --recycling 1 \
  --path $checkpoint \
  --log-interval 10 \
  --log-format simple \
  --required-batch-size-multiple 1

echo "=== INFERENCE DONE ==="
