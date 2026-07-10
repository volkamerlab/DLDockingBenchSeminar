#!/bin/bash
#Irem_Dogruoglu_7061348_Lizzie_Schmitz_7056551
#References:
#1. [https://stackoverflow.com/questions/60303997/activating-conda-environment-from-bash-script](https://stackoverflow.com/questions/60303997/activating-conda-environment-from-bash-script)
#2. [https://github.com/gnina/scripts](https://github.com/gnina/scripts)
#3. [https://stackoverflow.com/questions/47932165/tar-gz-unpacking-file-data](https://stackoverflow.com/questions/47932165/tar-gz-unpacking-file-data)
#4. [https://stackoverflow.com/questions/8033857/tar-archiving-that-takes-input-from-a-list-of-files](https://stackoverflow.com/questions/8033857/tar-archiving-that-takes-input-from-a-list-of-files)
#
#run_gnina.sh file runs the GNINA-Torch pipeline inside a container by following the workflow below:
#1) activating conda environment.
#2) unpacking data while doing checks on gninatorch.
#3) activating Python pipeline, and generating results.
#It takes gninatorch_env, full_data.tar.gz archive folder, and run_gnina_train.py script as inputs.
#Then it produces gninatorch_training, full_sealed_train.types, and proto_test_retrained files.

#!/usr/bin/env bash
set -e

echo "=== ENTERED run_gnina.sh ==="
echo "Working directory: $(pwd)"

#Activating conda env if it exists.
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate gninatorch_env
    export BABEL_DATADIR=/opt/conda/envs/gninatorch_env/share/openbabel/3.1.0
    echo "BABEL_DATADIR=$BABEL_DATADIR"
    echo "Using conda env:"
    conda info --envs | grep '*' || echo "unknown env"
else
    echo "WARNING: conda.sh not found, continuing without gninatorch_env"
fi

#Doing checks of python and gninatorch for further steps.
echo "Checking Python and GNINA-Torch..."
which python3
python3 -c "import sys; print(sys.executable)"
python3 -c "import gninatorch; print('gninatorch:', gninatorch.__file__)"
python3 -m gninatorch.training --help || exit 1

#Unpacking the data with .tar.gz, not unzip.
echo "Unpacking train/validation archive..."
tar -xzf full_data.tar.gz


#Making sure directories for results exist.
#Current goal: retraining models on the full dataset and exporting the .pt files.
mkdir -p results/full_sealed_train_redocked \
         results/full_sealed_val_redocked \
         results/full_sealed_test_retrained \
         results/gninatorch_training \
         results/plots


#Running the current pipeline:
#1. Docking full_train_data and full_val_data.
#2. Generating .types for val and train.
#3. Retraining GNINA-Torch models on the full training dataset and val dataset.
#4. Generating training plots.

python3 run_gnina_train.py \
  --train-csv full_sealed_train.csv \
  --train-input-dir full_sealed_train \
  --train-out-dir results/full_sealed_train_redocked \
  --val-csv full_sealed_val.csv \
  --val-input-dir full_sealed_val \
  --val-out-dir results/full_sealed_val_redocked \
  --run \
  --epochs 5 \
  --make-types \
  --train-types full_sealed_train.types \
  --val-types full_sealed_val.types \
  --train-model \
  --models default2018 dense \
  --seeds 1 \
  --make-plots \
  --plot-dir results/plots

#ADDED FOR VALIDATION:
#  --val-csv data/full_sealed_val.csv \
#  --val-input-dir data/full_sealed_val \
#  --val-out-dir results/full_sealed_val_redocked \
#  --val-types data/full_sealed_val.types \

#ADD LATER FOR REDOCKING: 
#   --test-csv data/full_sealed_test.csv \
#  --test-input-dir data/full_sealed_test \
#  --test-out-dir results/full_sealed_test_retrained \

status=$?
echo "Python/GNINA exit status: $status"

#Packaging the retrained checkpoint outputs for submission.
echo "Compressing retrained checkpoints into tarball..."

tar -czf results/gninatorch_training/retrained_checkpoints.tar.gz \
  -C results/gninatorch_training \
  --exclude="retrained_checkpoints.tar.gz" \
  .


exit $status
