#!/bin/bash
# Install missing dependencies to the container's user space
pip install --user seaborn scikit-learn safetensors tqdm pandas transformers

export HF_TOKEN=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
export WANDB_API_KEY=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

# Add the local install path to Python's search path
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages

# Execute your script
python3 task3_analysis.py