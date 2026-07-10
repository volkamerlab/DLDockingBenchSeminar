#!/bin/bash
BASE="/home/bdldt_team007/DLDockingBenchSeminar/run6"
data_path="$BASE/data"
results_path="$BASE/inference_results"
checkpoint="$BASE/checkpoints/checkpoint_best.pt"

mkdir -p $results_path

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

echo "=== INFERENCE (run6 checkpoint_best) START ==="

python /home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/unimol/infer.py \
  --user-dir /home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/unimol \
  $data_path \
  --valid-subset valid \
  --results-path $results_path \
  --num-workers 0 \
  --ddp-backend=c10d \
  --batch-size 8 \
  --task docking_pose_v2 \
  --loss docking_pose_v2 \
  --arch docking_pose_v2 \
  --conf-size 10 \
  --dist-threshold 8.0 \
  --recycling 1 \
  --path $checkpoint \
  --log-interval 50 \
  --log-format simple \
  --required-batch-size-multiple 1

echo "=== INFERENCE (run6 checkpoint_best) DONE ==="
