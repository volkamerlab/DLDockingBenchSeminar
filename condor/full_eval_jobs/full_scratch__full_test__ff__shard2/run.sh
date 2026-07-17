#!/bin/bash
set -euo pipefail
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}
python3 /home/bdldt_team002/eval_full_data/evaluation.py --dataset full_test --output_csv /home/bdldt_team002/eval_full_data/out/full_scratch__full_test__ff__shard2.csv --top_n 1 --shard_idx 2 --num_shards 6
