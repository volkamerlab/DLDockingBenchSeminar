#!/bin/bash
#Irem_Dogruoglu_7061348_Lizzie_Schmitz_7056551
#References:
#1. https://stackoverflow.com/questions/60303997/activating-conda-environment-from-bash-script
#2. https://github.com/gnina/scripts
#3. https://stackoverflow.com/questions/47932165/tar-gz-unpacking-file-data
#4. https://stackoverflow.com/questions/8033857/tar-archiving-that-takes-input-from-a-list-of-files
# run_gnina.sh file run the GNINA-Torch pipeline inside a container by following the workflow on below:
# 1) activating conda environment.
# 2) unpacking data while doing checks on gninatorch.
# 3) activating Python pipeline, and generating results.
# It takes gninatorch_env, prototype_data.tar.gz as archive folder and run_proto_gnina.py script.
# Then it produces gninatorch_training, plots results, data/proto_train.types, and proto_test_retrained files.

set -e

echo "=== ENTERED run_gnina.sh ==="
echo "Working directory: $(pwd)"

#activating conda env if it exists.
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

#doing checks of python and gninatorch for further steps.
echo "Checking Python and GNINA-Torch..."
which python3
python3 -c "import sys; print(sys.executable)"
python3 -c "import gninatorch; print('gninatorch:', gninatorch.__file__)"
python3 -m gninatorch.training --help || exit 1

#unpacking the data with .tar.gz, not unzip.
echo "Unpacking prototype_data.tar.gz..."
tar -xzf prototype_data.tar.gz

#making sure directories for results exist.
mkdir -p results/proto_test_retrained \
         results/gninatorch_training \
         results/plots

#running the all pipeline in the workflow of: docking, generating types, training, redocking and generating plots.
# LATER: add for validation set: --val-types data/proto_test.types 
python3 run_proto_gnina.py \
  --train-csv data/proto_train.csv \
  --train-input-dir data/proto_train \
  --train-out-dir results/proto_train \
  --csv data/proto_test.csv \
  --input-dir data/proto_test \
  --retrained-test-out-dir results/proto_test_retrained \
  --run \
  --epochs 5 \
  --make-types \
  --train-types data/proto_train.types \
  --train-model \
  --models default2018 dense \
  --seeds 1 \
  --make-plots \
  --plot-dir results/plots

# ADD LATER FOR VALIDATION: 
  # --val-csv data/validation.csv \
  # --val-input-dir data/validation \
  # --val-types data/validation.types \

status=$?
echo "Python/GNINA exit status: $status"

# Output retrained proto_test outputs as a .tar.gz file:
echo "Compressing retrained proto_test outputs into tarball..."

tar -czf results/proto_test_retrained/proto_test_retrained_outputs.tar.gz \
  -C results/proto_test_retrained \
  --exclude="proto_test_retrained_outputs.tar.gz" \
  .

exit $status
