#!/bin/bash
set -euo pipefail
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}
python3 /home/bdldt_team002/eval_full_data/evaluation.py --dataset posebusters_filtered --output_csv /home/bdldt_team002/eval_full_data/out/full_scratch__posebusters_filtered__uncorrected.csv --top_n 1 --shard_idx 0 --num_shards 1
