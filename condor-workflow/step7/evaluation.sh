#!/bin/bash

source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base

cd /home/bdldt_team001/DLDockingBenchSeminar

python scripts/csv_file.py

#python evaluation/evaluation.py --dataset full_test --output_csv results/my_eval_fullset.csv
#python evaluation/evaluation.py --dataset posebusters_filtered --output_csv results/my_eval_posebusters.csv
