#!/usr/bin/env bash
# run_infer.sh <dataset> <model_file> <tag>
# -----------------------------------------------------------------------------
# Dock <dataset> with KarmaDock and export the predicted poses in ALL THREE
# post-processing variants the official KarmaDock pipeline provides:
#   uncorrected (raw network output) · FF-corrected · align-corrected.
# This step produces POSES ONLY. Scoring is a separate step (see evaluate.sh /
# README: `python evaluation/evaluation.py --dataset proto_test`).
#
#   <dataset>     test set present in the CWD as <dataset>.csv + <dataset>/
#   <model_file>  /app/KarmaDock/trained_models/karmadock_screening.pkl  (P1, in image)
#                 or a transferred <basename>.pkl                         (P2/P3)
#   <tag>         output label, e.g. p2_scratch
#
# Writes (relative to the submit dir):
#   results/<tag>/<dataset>         uncorrected poses   (<id>_pred.sdf, best-first)
#   results/<tag>/<dataset>_ff      FF-corrected poses
#   results/<tag>/<dataset>_align   align-corrected poses
#
# Portable: uses only /app/* (baked image) and ./scripts ./<dataset> (transferred).
# Deterministic: same image + weights + --random_seed 2023 => identical poses.
set -euo pipefail
set -x

DATASET=$1
MODEL=$2
TAG=$3

SUBMIT="$PWD"
# A transferred checkpoint arrives as a bare basename; resolve it before we cd into
# /app/KarmaDock/utils, where a relative path would not be found.
case "$MODEL" in /*) : ;; *) MODEL="$SUBMIT/$MODEL" ;; esac

KD=/app/KarmaDock
WORK="$(mktemp -d -t kd-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
KIN="$WORK/kin"; GRAPH="$WORK/graphs"; KDOUT="$WORK/kdout"
mkdir -p "$KIN" "$GRAPH" "$KDOUT"

echo "=== [$TAG] dock $DATASET with $MODEL ==="

# 1) seminar layout -> KarmaDock layout
python3 "$SUBMIT/scripts/convert_seminar_to_karmadock.py" \
    --csv "$SUBMIT/$DATASET.csv" --src_dir "$SUBMIT/$DATASET" --out_dir "$KIN"

# 2) pocket extraction (CPU, prody) + 3) graph generation
cd "$KD/utils"
python3 -u pre_processing.py  --complex_file_dir "$KIN"
python3 -u generate_graph.py  --complex_file_dir "$KIN" --graph_file_dir "$GRAPH"
echo "=== [$TAG] graphs built: $(ls "$GRAPH" | wc -l) ==="

# 4) dock + score + correct (correct True writes the uncorrected, FF and align SDFs)
python3 -u ligand_docking.py \
    --graph_file_dir "$GRAPH" \
    --model_file "$MODEL" \
    --out_dir "$KDOUT" \
    --docking True --scoring True --correct True \
    --batch_size 64 --random_seed 2023

# 5) export each pose variant to the seminar layout (best-pose-first)
#    mode -> output-dir suffix
export_variant () {
    local mode=$1 suffix=$2
    local out="$SUBMIT/results/$TAG/${DATASET}${suffix}"
    mkdir -p "$out"
    python3 "$SUBMIT/scripts/convert_karmadock_to_seminar.py" \
        --input_dir "$KDOUT" --csv "$SUBMIT/$DATASET.csv" --out_dir "$out" --mode "$mode"
    echo "=== [$TAG] ${mode}: $(ls "$out"/*_pred.sdf 2>/dev/null | wc -l) poses -> ${out#$SUBMIT/} ==="
}
export_variant uncorrected     ""
export_variant ff_corrected    "_ff"
export_variant align_corrected "_align"

echo "=== [$TAG] inference done (3 pose variants in results/$TAG/) ==="
