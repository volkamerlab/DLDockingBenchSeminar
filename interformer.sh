#!/bin/bash
# Install missing dependencies to the container's user space
pip install --user seaborn scikit-learn safetensors tqdm pandas transformers

# Add the local install path to Python's search path
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages


python3 -U train.py