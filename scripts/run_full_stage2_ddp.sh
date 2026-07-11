#!/usr/bin/env bash
# run_full_stage2_ddp.sh — ISOLATED multi-GPU (DDP) Stage-2 TEST on the full dataset.
# -----------------------------------------------------------------------------
# Standalone experiment, fully separated from the prototype submission. It:
#   * READS (read-only) the already-built full-data graphs + a Stage-1 checkpoint snapshot,
#   * launches train_ddp.py (Stage-2 docking only) across ALL GPUs on the node via torchrun
#     (DistributedDataParallel); a single GPU falls back to plain python,
#   * WRITES checkpoints/logs/W&B to its OWN dir, so nothing in the main run/results is touched.
#
# NOTHING here writes into ~/repro_test/work_full/ckpt (the main run's p_stage1/p_stage2) or
# into the DLDockingBenchSeminar submission. Inputs are opened read-only.
#
# Effective batch = NGPU * batch_size * accum_steps  (defaults keep it 64).
set -euo pipefail
set -x

KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}
SUBMIT="$PWD"                      # job sandbox; train_ddp.py + seminar_csv.py land here

# --- paths (override via the sub's `environment` if relocated) ---
DATA="${FULL_DATA_DIR:-$HOME/repro_test/full_data}"          # full_train.csv / full_val.csv (read-only)
GRAPHS="${FULL_GRAPHS_DIR:-$HOME/repro_test/work_full}"      # shared .dgl graphs (READ-ONLY)
S2WORK="${S2_WORK_DIR:-$HOME/stage2_mgpu_test/work}"         # ISOLATED output (own dir)
STAGE1_CKPT="${STAGE1_CKPT:-$HOME/stage2_mgpu_test/stage1_snapshot.pkl}"   # frozen Stage-1 weights
BS="${S2_BATCH:-8}"; ACC="${S2_ACCUM:-2}"                    # per-GPU batch + accumulation
EP="${S2_EPOCHS:-1000}"                                      # smoke test sets this small (e.g. 3)

mkdir -p "$S2WORK"
[ -f "$STAGE1_CKPT" ] || { echo "ERROR: Stage-1 checkpoint not found: $STAGE1_CKPT"; exit 3; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1
RUN="full_stage2_ddp_$(date +%Y%m%d-%H%M%S)"                 # unique W&B run name per launch
echo "=== Stage-2 DDP on $NGPU GPU(s); effective batch = $((NGPU*BS*ACC)); run=$RUN ==="

ARGS=( "$SUBMIT/train_ddp.py"
  --csv "$DATA/full_train.csv"   --graph_dir "$GRAPHS/full_train/graphs" --complex_dir "$GRAPHS/full_train/complex"
  --val_csv "$DATA/full_val.csv" --val_graph_dir "$GRAPHS/full_val/graphs"
  --out_dir "$S2WORK/ckpt_s2_ddp" --init_model "$STAGE1_CKPT" --pos_r 1
  --lr 1e-4 --weight_decay 1e-4 --patience 20 --epochs "$EP"
  --batch_size "$BS" --accum_steps "$ACC" --num_workers 3 --random_seed 42 --resume
  --wandb --wandb_project karmadock-seminar --wandb_run_name "$RUN" )

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "${ARGS[@]}"
else
  python3 -u "${ARGS[@]}"
fi
echo "=== Stage-2 DDP done: $S2WORK/ckpt_s2_ddp/karmadock_team002.pkl ==="
