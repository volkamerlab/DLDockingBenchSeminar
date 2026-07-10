export PYTHONNOUSERSITE=1

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export DISABLE_WANDB=true
export WANDB_SILENT=true

export PYTHONPATH=/home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2:$PYTHONPATH

cd /home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2

bash train.sh

