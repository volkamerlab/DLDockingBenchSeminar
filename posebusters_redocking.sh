#!/usr/bin/env bash
#Irem_Dogruoglu_7061348_Lizzie_Schmitz_7056551
#References:
#1.https://stackoverflow.com/questions/60303997/activating-conda-environment-from-bash-script
#2.https://github.com/gnina/scripts
#3.https://stackoverflow.com/questions/47932165/tar-gz-unpacking-file-data
#4.https://stackoverflow.com/questions/8033857/tar-archiving-that-takes-input-from-a-list-of-files
#5 https://github.com/degrado-lab/PoseBusters-Benchmark
#
# redocking.sh file runs the GNINA redocking pipeline inside a container by following the workflow below:
# 1) activating conda environment.
# 2) unpacking posebusters data while doing checks on gninatorch / gnina.
# 3) activating Python redocking pipeline, and generating results.
# It takes gninatorch_env, posebusters_filtered.tar.gz, retrained checkpoint outputs that are in retrained_checkpoints.tar.gz, and posebusters_redocking.py script.
# Then it produces full_sealed_test_retrained files.

set -e

echo "=== ENTERED redocking_posebusters.sh ==="
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

#Doing checks of python and gninatorch for further steps.
echo "Checking Python and GNINA-Torch..."
which python3
python3 -c "import sys; print(sys.executable)"
python3 -c "import gninatorch; print('gninatorch:', gninatorch.__file__)"
gnina --help >/dev/null 2>&1 || exit 1

#Unpacking the PoseBusters data archive.
echo "Unpacking posebusters_filtered.tar.gz..."
tar -xzf posebusters_filtered.tar.gz

#Unpacking retrained model checkpoints.
echo "Unpacking retrained checkpoints..."
mkdir -p results/gninatorch_training
tar -xzf retrained_checkpoints.tar.gz -C results/gninatorch_training

#Checking if the result directories exist.
mkdir -p results/posebusters_redocked \
         results/gninatorch_training

# Running the redocking workflow:
# 1) redocking PoseBusters with default2018 retrained checkpoint
# 2) redocking PoseBusters with dense retrained checkpoint
set +e
python3 posebusters_redocking.py \
  --csv data/posebusters_filtered.csv \
  --input-dir data/posebusters_filtered \
  --training-out-dir results/gninatorch_training/default2018_seed1 \
  --retrained-test-out-dir results/posebusters_redocked/default2018_seed1 \
  --run
status1=$?
python3 posebusters_redocking.py \
  --csv data/posebusters_filtered.csv \
  --input-dir data/posebusters_filtered \
  --training-out-dir results/gninatorch_training/dense_seed1 \
  --retrained-test-out-dir results/posebusters_redocked/dense_seed1 \
  --run
status2=$?
set -e
status=$((status1 || status2))
echo "Python/GNINA exit status: $status"

#Gathering retrained PoseBusters outputs as a .tar.gz file.
echo "Compressing outputs into tarball..."

tar -czf results_posebusters_redocking.tar.gz \
  results/posebusters_redocked

exit $status
