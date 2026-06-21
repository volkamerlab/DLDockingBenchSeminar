#!/usr/bin/env bash
# run_full_train.sh <scratch|finetune>
# -----------------------------------------------------------------------------
# FULL-dataset training (full_train 23,483 / full_val 2,609) on one GPU.
#
# Unlike run_train.sh (prototype, 712 complexes transferred into the job), the full
# data is ~14 GB, so it is NOT transferred: it is read from the mounted home
# (+WantGPUHomeMounted), and graphs + checkpoints are written to a persistent home
# work dir so a force-reschedule can --resume without redoing the (long) preprocess.
#
# Layout expected on the mounted home (override the root with FULL_DATA_DIR):
#   $FULL_DATA_DIR/full_train.csv  + full_train/<id>_{ligand_refined.sdf,protein_refined.pdb}
#   $FULL_DATA_DIR/full_val.csv    + full_val/...
#
# Outputs (persistent, on the home):
#   scratch  -> $FULL_WORK_DIR/ckpt/p_stage2/karmadock_team002.pkl
#   finetune -> $FULL_WORK_DIR/ckpt/p_finetune/karmadock_team002.pkl
#
# The full train/val split is curated (full_val is NOT a random carve-out), so we
# pass it via --val_csv / --val_graph_dir (train.py + seminar_csv.py handle the
# full-metadata CSV schema directly -- no data edits needed).
set -euo pipefail
set -x

MODE=${1:?usage: run_full_train.sh <scratch|finetune>}
KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}

DATA="${FULL_DATA_DIR:-$HOME/repro_test/full_data}"
WORK="${FULL_WORK_DIR:-$HOME/repro_test/work_full}"
# HTCondor flattens the executable to the sandbox root but lands transfer_input_files=scripts
# in a scripts/ subdir, so the helper code is at $PWD/scripts (NOT dirname "$0", which is the root).
SUBMIT="$PWD"; SCRIPTS="$SUBMIT/scripts"
CKPT="$WORK/ckpt"; mkdir -p "$CKPT"

# --- preprocess one split once (seminar layout -> KarmaDock layout, pockets, graphs) ---
preprocess() {                               # $1 = split name (full_train | full_val)
  local split="$1"
  local kin="$WORK/$split/complex" graph="$WORK/$split/graphs"
  if [ -f "$WORK/$split/.preprocessed" ]; then echo "# $split already preprocessed"; return; fi
  mkdir -p "$kin" "$graph"
  python3 "$SCRIPTS/convert_seminar_to_karmadock.py" \
      --csv "$DATA/$split.csv" --src_dir "$DATA/$split" --out_dir "$kin"
  ( cd "$KD/utils"
    python3 -u pre_processing.py --complex_file_dir "$kin"
    python3 -u generate_graph.py --complex_file_dir "$kin" --graph_file_dir "$graph" )
  touch "$WORK/$split/.preprocessed"
}
preprocess full_train
preprocess full_val

COMMON=( --csv "$DATA/full_train.csv" --graph_dir "$WORK/full_train/graphs"
         --complex_dir "$WORK/full_train/complex"
         --val_csv "$DATA/full_val.csv" --val_graph_dir "$WORK/full_val/graphs"
         --batch_size 4 --accum_steps 16 --random_seed 42 --resume )

if [ "$MODE" = "scratch" ]; then
  # Stage 1 - scoring / MDN only (pos_r 0)
  if [ ! -f "$CKPT/p_stage1/stage.done" ]; then
    python3 -u "$SCRIPTS/train.py" "${COMMON[@]}" \
        --out_dir "$CKPT/p_stage1" --init_model "" --pos_r 0 \
        --lr 1e-3 --weight_decay 1e-5 --patience 70 --epochs 1000
    touch "$CKPT/p_stage1/stage.done"
  fi
  # Stage 2 - docking + scoring (pos_r 1), init from Stage-1 best
  python3 -u "$SCRIPTS/train.py" "${COMMON[@]}" \
      --out_dir "$CKPT/p_stage2" --init_model "$CKPT/p_stage1/karmadock_team002.pkl" \
      --pos_r 1 --lr 1e-4 --weight_decay 1e-4 --patience 20 --epochs 1000 --jitter 0.05
  echo "=== FULL from-scratch done: $CKPT/p_stage2/karmadock_team002.pkl ==="
elif [ "$MODE" = "finetune" ]; then
  python3 -u "$SCRIPTS/train.py" "${COMMON[@]}" \
      --out_dir "$CKPT/p_finetune" --init_model "$KD/trained_models/karmadock_screening.pkl" \
      --pos_r 1 --lr 1e-4 --weight_decay 0 --patience 30 --epochs 500
  echo "=== FULL fine-tune done: $CKPT/p_finetune/karmadock_team002.pkl ==="
else
  echo "ERROR: mode must be 'scratch' or 'finetune'"; exit 2
fi
