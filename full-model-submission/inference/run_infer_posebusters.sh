export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2:$PYTHONPATH
BASE="/home/bdldt_team007/DLDockingBenchSeminar"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "=== POSEBUSTERS INFERENCE START ==="
python $BASE/unimol/unimol_docking_v2/unimol/infer.py \
  --user-dir $BASE/unimol/unimol_docking_v2/unimol \
  $BASE/data/processed_posebusters \
  --valid-subset posebusters_filtered \
  --results-path $BASE/run6/posebusters_results \
  --num-workers 0 \
  --ddp-backend=c10d \
  --batch-size 8 \
  --task docking_pose_v2 \
  --loss docking_pose_v2 \
  --arch docking_pose_v2 \
  --conf-size 10 \
  --dist-threshold 8.0 \
  --recycling 1 \
  --path $BASE/run6/checkpoints/checkpoint_best.pt \
  --log-interval 50 \
  --log-format simple \
  --required-batch-size-multiple 1
echo "=== POSEBUSTERS INFERENCE DONE ==="
