# **Benchmarking DL-based Docking Tools** — Summer Semester 2026
  
# DiffDock-Pocket — Prototype Submission

**Team 4 — DiffDock-Pocket: Dhanya, Divyashree**

## Overview
This branch contains our team's prototype submission for benchmarking **DiffDock-Pocket** as part of the DLDockingBenchSeminar (SS26).

## Repository structure
training/

├── train.py                  — training script

├── checkpoint/                — our trained checkpoint (best_ema_model.pt, model_parameters.yml)

├── diffdock_training.sub      — Condor submit file for training

└── run_training.sh            — training run script
inference/

├── inference.py                — inference script

├── diffdock_inference.sub      — Condor submit file for inference

├── run_inference.sh            — inference run script

└── (additional inference run artifacts)
datasets/        — data loading and preprocessing code (shared by training and inference)

models/           — DiffDock-Pocket model architecture

utils/            — shared helper functions (training loop, sampling, parsing, etc.)

evaluation/       — RMSD evaluation script (evaluation.py), provided by seminar instructors

data/             — prototype training/test datasets (proto_train, proto_test)

results/          — evaluation results, predicted poses, and results notebook

environment.yml   — conda environment specification

evaluate_files.py — alternate evaluation script

## Docker Image
We built and use a custom Docker image based on `silterra/diffdock-pocket`, with fixes for HPC compatibility:
docker.io/dhanya24/diffdock-pocket-v8:prototype
