#!/usr/bin/env bash
# evaluate.sh — the scoring step (separate from docking).
# -----------------------------------------------------------------------------
# Runs the seminar's official evaluator over the predicted poses. For each pipeline
# and correction variant it runs the documented command
#   python evaluation/evaluation.py --dataset proto_test
# and writes results/<pipeline>_<variant>_evaluation.csv, plus the headline
#   results/proto_test_evaluation.csv   (= the from-scratch P2 uncorrected poses).
#
# Run from the repo ROOT, in an environment with the seminar deps (RDKit) — e.g.
# inside the docker image, or any env where `python evaluation/evaluation.py` works.
# Reference structures come from the unzipped bundle (data/prototype_model_data/).
set -euo pipefail

ROOT="$PWD"
EVALPY="$ROOT/evaluation/evaluation.py"

# Resolve the CSV + reference dir whether we run from the repo root (data/...) or inside
# a flattened HTCondor sandbox (transferred files arrive as basenames in the CWD).
first_existing () {   # <test-flag -f|-d> <candidate>...
    local flag=$1; shift
    local p; for p in "$@"; do [ "$flag" "$p" ] && { echo "$p"; return; }; done
}
CSV=$(first_existing -f "$ROOT/data/proto_test.csv" "$ROOT/proto_test.csv")
REF=$(first_existing -d "$ROOT/data/prototype_model_data/proto_test" "$ROOT/data/proto_test" "$ROOT/proto_test")

[ -n "$CSV" ]    || { echo "ERROR: proto_test.csv not found (looked in data/ and CWD)"; exit 1; }
[ -n "$REF" ]    || { echo "ERROR: proto_test reference dir not found — unzip data/prototype_model_data.zip -d data/prototype_model_data"; exit 1; }
[ -f "$EVALPY" ] || { echo "ERROR: missing $EVALPY"; exit 1; }

# evaluation.py reads results/proto_test/ for a fixed dataset name, so we point it at
# each pose set through a throwaway workspace of symlinks (the poses are never moved).
eval_one () {   # <poses_dir> <out_csv>
    local poses=$1 out=$2
    local E; E="$(mktemp -d)"
    mkdir -p "$E/data" "$E/results"
    ln -sf "$CSV"   "$E/data/proto_test.csv"
    ln -sf "$REF"   "$E/data/proto_test"
    ln -sf "$poses" "$E/results/proto_test"
    ( cd "$E" && python3 "$EVALPY" --dataset proto_test --no_pb_valid --output_csv "$out" )
    rm -rf "$E"
}

for tag in p1_baseline p2_scratch p3_finetune; do
  for variant in ":uncorrected" "_ff:ff" "_align:align"; do
    suffix="${variant%%:*}"; label="${variant##*:}"
    poses="$ROOT/results/$tag/proto_test${suffix}"
    if [ ! -d "$poses" ]; then echo "skip $tag/$label (no poses at ${poses#$ROOT/})"; continue; fi
    eval_one "$poses" "$ROOT/results/${tag}_${label}_evaluation.csv"
    echo "wrote results/${tag}_${label}_evaluation.csv"
  done
done

# headline = the from-scratch (P2) uncorrected poses, named for the default
# `python evaluation/evaluation.py --dataset proto_test` invocation.
if [ -f "$ROOT/results/p2_scratch_uncorrected_evaluation.csv" ]; then
  cp "$ROOT/results/p2_scratch_uncorrected_evaluation.csv" "$ROOT/results/proto_test_evaluation.csv"
  echo "wrote results/proto_test_evaluation.csv (= P2 from-scratch, uncorrected)"
fi
