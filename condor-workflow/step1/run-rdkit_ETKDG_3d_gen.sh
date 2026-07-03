#!/bin/bash
source /main/home/mambaforge/etc/profile.d/conda.sh
conda activate base

cd /home/bdldt_team001/DLDockingBenchSeminar

cd /home/bdldt_team001/DLDockingBenchSeminar
python tools/rdkit_ETKDG_3d_gen.py data/proto_val/ligand/rcsb/ data/proto_val/uff/