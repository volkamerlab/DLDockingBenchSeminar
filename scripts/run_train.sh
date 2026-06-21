#!/usr/bin/env bash
# run_train.sh <scratch|finetune>
# -----------------------------------------------------------------------------
# SELF-CONTAINED training on proto_train (712), portable (no machine-specific paths).
# Preprocesses the train set, then runs train.py with the paper's hyper-parameters.
# This is the LONG path (preprocess + several hours on one GPU) and is the BONUS;
# the graded reproduction is the docking + scoring path (run_infer.sh + evaluate.sh).
#
# Output checkpoints (transferred back) land in ./ckpt/:
#   scratch  -> ckpt/p2_stage2/karmadock_team002.pkl   (== model/p2_scratch_karmadock_team002.pkl)
#   finetune -> ckpt/p3_finetune/karmadock_team002.pkl (== model/p3_finetune_karmadock_team002.pkl)
#
# Expects proto_train.csv + proto_train/ in the CWD (unzip data/prototype_model_data.zip).
# Effective batch = 64 (batch_size 4 * accum_steps 16); split val_frac 0.1, seed 42.
set -euo pipefail
set -x

MODE=${1:?usage: run_train.sh <scratch|finetune>}
SUBMIT="$PWD"
KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}

# Persistent work dir (NOT mktemp) so --resume survives a force-reschedule.
WORK="$SUBMIT/work_train"; KIN="$WORK/complex"; GRAPH="$WORK/graphs"
mkdir -p "$KIN" "$GRAPH" "$SUBMIT/ckpt"

# --- preprocess proto_train: seminar -> KarmaDock layout, pockets, graphs ---
if [ ! -f "$WORK/.preprocessed" ]; then
  python3 "$SUBMIT/scripts/convert_seminar_to_karmadock.py" \
      --csv "$SUBMIT/proto_train.csv" --src_dir "$SUBMIT/proto_train" --out_dir "$KIN"
  ( cd "$KD/utils"
    python3 -u pre_processing.py --complex_file_dir "$KIN"
    python3 -u generate_graph.py --complex_file_dir "$KIN" --graph_file_dir "$GRAPH" )
  touch "$WORK/.preprocessed"
fi

cd "$SUBMIT"
if [ "$MODE" = "scratch" ]; then
  # Stage 1 — scoring / MDN only (pos_r 0)
  if [ ! -f ckpt/p2_stage1/stage.done ]; then
    python3 -u scripts/train.py --csv proto_train.csv --graph_dir "$GRAPH" --complex_dir "$KIN" \
        --out_dir ckpt/p2_stage1 --init_model "" --pos_r 0 --lr 1e-3 --weight_decay 1e-5 \
        --batch_size 4 --accum_steps 16 --patience 70 --epochs 1000 --val_frac 0.1 --random_seed 42 --resume
    touch ckpt/p2_stage1/stage.done
  fi
  # Stage 2 — docking + scoring (pos_r 1), init from Stage-1 best
  python3 -u scripts/train.py --csv proto_train.csv --graph_dir "$GRAPH" --complex_dir "$KIN" \
      --out_dir ckpt/p2_stage2 --init_model ckpt/p2_stage1/karmadock_team002.pkl --pos_r 1 \
      --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --accum_steps 16 --patience 20 --epochs 1000 \
      --val_frac 0.1 --random_seed 42 --jitter 0.05 --resume
  echo "=== P2 from-scratch done: ckpt/p2_stage2/karmadock_team002.pkl ==="
elif [ "$MODE" = "finetune" ]; then
  python3 -u scripts/train.py --csv proto_train.csv --graph_dir "$GRAPH" --complex_dir "$KIN" \
      --out_dir ckpt/p3_finetune --init_model /app/KarmaDock/trained_models/karmadock_screening.pkl \
      --pos_r 1 --lr 1e-4 --weight_decay 0 --batch_size 4 --accum_steps 16 --patience 30 --epochs 500 \
      --val_frac 0.1 --random_seed 42 --resume
  echo "=== P3 fine-tune done: ckpt/p3_finetune/karmadock_team002.pkl ==="
else
  echo "ERROR: mode must be 'scratch' or 'finetune'"; exit 2
fi
