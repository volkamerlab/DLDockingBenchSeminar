#!/bin/bash
# Per-shard InterFormer docking. Runs find (docking) + stat (summary) on ONE
# shard's subset, then packages its output for transfer back. The experimental-
# label merge/label step is intentionally NOT here -- it runs once, after all
# shards finish, in merge_shards.py (that missing 'data/proto_train_val_final.csv'
# is exactly what crashed the previous run).
set -uo pipefail

PROJECT_ROOT="$(pwd)"
SHARD="${SHARD:-${1:-0}}"
echo "=== Shard ${SHARD} running in: ${PROJECT_ROOT} ==="
ls -F   # debug: what actually got transferred into the sandbox

DOCK_FOLDER="dock_results/energy_train"
mkdir -p "${DOCK_FOLDER}" logs

# --- locate + unpack the four input tarballs, WHEREVER HTCondor placed them ---
# The old script looked only in ./ but the tarballs landed under
# dock_results/energy_train/, so the unpack loop never fired. Searching the
# sandbox makes this robust to HTCondor's path handling.
unpacked=0
for name in complex gaussian_predict ligand uff; do
    tarball="$(find . -maxdepth 3 -name "${name}.tar.gz" -print -quit 2>/dev/null)"
    if [ -n "${tarball}" ] && [ -f "${tarball}" ]; then
        echo "=== Unpacking ${tarball} -> ${DOCK_FOLDER} ==="
        if tar -xzf "${tarball}" -C "${DOCK_FOLDER}"; then
            rm -f "${tarball}"          # free the compressed copy immediately
            unpacked=$((unpacked + 1))
        else
            echo "ERROR: failed to extract ${tarball}" >&2
        fi
    else
        echo "WARNING: ${name}.tar.gz not found in sandbox" >&2
    fi
done
if [ "${unpacked}" -eq 0 ]; then
    echo "FATAL: no input tarballs were unpacked; aborting shard ${SHARD}." >&2
    exit 1
fi
echo "=== Contents of ${DOCK_FOLDER} after unpacking ==="
ls -F "${DOCK_FOLDER}"

# --- PyVina ---
echo "=== Installing PyVina ==="
pip install --user --no-cache-dir ./docking || { echo "FATAL: PyVina install failed" >&2; exit 1; }

export OMP_NUM_THREADS="32,32"

# --- dock (find), then summarize (stat). No CSV needed for docking. -----------
# If you want to dock from the UFF conformers, add: --uff_folder uff
echo "=== reconstruct: find (docking) ==="
python docking/reconstruct_ligands.py -y --cwd "${DOCK_FOLDER}" -y --find_all find \
    || { echo "FATAL: reconstruct find failed" >&2; exit 1; }

echo "=== reconstruct: stat (summary) ==="
python docking/reconstruct_ligands.py --cwd "${DOCK_FOLDER}" --find_all stat \
    || { echo "FATAL: reconstruct stat failed" >&2; exit 1; }

# --- package this shard's output for transfer back ---------------------------
OUT="out_shard_${SHARD}.tar.gz"
if [ -d "${DOCK_FOLDER}/ligand_reconstructing" ]; then
    echo "=== Packaging ${DOCK_FOLDER}/ligand_reconstructing -> ${OUT} ==="
    tar -czf "${OUT}" -C "${DOCK_FOLDER}" ligand_reconstructing \
        || { echo "FATAL: could not package output" >&2; exit 1; }
else
    echo "FATAL: ${DOCK_FOLDER}/ligand_reconstructing missing; nothing to return." >&2
    exit 1
fi

n_sdf=$(find "${DOCK_FOLDER}/ligand_reconstructing" -name '*.sdf' | wc -l)
echo "=== Shard ${SHARD} complete: ${n_sdf} SDF files produced ==="
