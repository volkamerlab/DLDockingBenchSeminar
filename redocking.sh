#!/usr/bin/env bash
#Irem_Dogruoglu_7061348_Lizzie_Schmitz_7056551
#References:
#1.https://stackoverflow.com/questions/60303997/activating-conda-environment-from-bash-script
#2.https://github.com/gnina/scripts
#3.https://stackoverflow.com/questions/47932165/tar-gz-unpacking-file-data
#4.https://stackoverflow.com/questions/8033857/tar-archiving-that-takes-input-from-a-list-of-files
#
# redocking.sh file runs the GNINA redocking pipeline by following the workflow below:
# 1)Activating conda environment.
# 2)Unpacking test data while doing checks on gninatorch and gnina.
# 3)Activating redocking pipeline, and generating results.
# It takes gninatorch_env, full_sealed_test.tar.gz, retrained checkpoint outputs that are in retrained_checkpoints.tar.gz, and redocking.py script.
# Then it produces full_sealed_test_retrained files.

set -e

echo "=== ENTERED redocking.sh ==="
echo "Working directory: $(pwd)"

# Activating conda env if it exists.
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

# Doing checks of python and gninatorch for further steps.
echo "Checking Python and GNINA-Torch..."
which python3
python3 -c "import sys; print(sys.executable)"
python3 -c "import gninatorch; print('gninatorch:', gninatorch.__file__)"
gnina --help >/dev/null 2>&1 || exit 1

# Unpacking the test data with .tar.gz, not unzip.
echo "Unpacking full_sealed_test.tar.gz..."
tar -xzf full_sealed_test.tar.gz

# Unpacking results of retraining data, in retrained_checkpoints.tar.gz
echo "Unpacking retrained checkpoints..."
mkdir -p results/gninatorch_training
tar -xzf retrained_checkpoints.tar.gz -C results/gninatorch_training

# Making sure directories for results exist.
mkdir -p results/full_sealed_test_retrained \
         results/gninatorch_training

# Running the redocking workflow:
# 1)Redocking full_sealed_test with retrained checkpoint.
# 2)Generating full_sealed_test_retrained outputs.


# Calling the redocking.py TWICE for both retrained models
python3 redocking.py \
  --csv data/full_sealed_test.csv \
  --input-dir data/full_sealed_test \
  --training-out-dir results/gninatorch_training/default2018_seed1 \
  --retrained-test-out-dir results/full_sealed_test_retrained/default2018_seed1 \
  --run

python3 redocking.py \
  --csv data/full_sealed_test.csv \
  --input-dir data/full_sealed_test \
  --training-out-dir results/gninatorch_training/dense_seed1 \
  --retrained-test-out-dir results/full_sealed_test_retrained/dense_seed1 \
  --run

status=$?
echo "Python/GNINA exit status: $status"

# Output retrained full_sealed_test outputs as a .tar.gz file.
echo "Compressing outputs into tarball..."

tar -czf results_redocking.tar.gz \
  results/full_sealed_test_retrained

exit $status
