# **Benchmarking DL-based Docking Tools: GNINA** 
  
## Background
 This prototype submission contains an implementation of retrained GNINA convolutional neural network (CNN) models that score docking of ligands to receptors, using the provided dataset for this seminar, in order to benchmark various DL Docking software tools across a standardized dataset, to more fairly benchmark tool performances. The pipeline abides by the GNINA version 1.3 training structure outlined in the literature by McNutt et. al 2025, by docking the provided training data (proto_train) with default GNINA to generate docking poses, preprocessing the training poses into GNINA-compatible .types files, retraining each GNINA 1.3 CNN model type (default2018 or dense) with a seed input, then docking the provided unseen test data on each retrained model type to generate docking evaluation metrics. 

 For future steps, for ensemble training, we will require 1 dense model and 2 knowledge-distilled student models (1 dense, 1 default2018) to form a GNINA version 1.3 ensemble. 
 
## About GNINA
 GNINA generates many ligand conformations with Monte Carlo sampling (MCMC), which are scored via the incorporated AutoDock VINA scoring function within its Deep Learning (DL) CNN architecture, for quick energy minimization. For each candidate docking pose, GNINA voxelizes the ligand-receptor complex via libmolgrid into a 3D atomic grid, carrying out two training tasks: scoring CNN pose score (the probability that the pose is correct) and CNN affinity (the binding affinity in pK units, such that pK = -log10(Kd)). With these two metrics, GNINA uses a CNN to carry out multi-task learning, such that pose score loss is derived from the cross entropy loss, determining whether the docking pose is ≤ 2 Å RMSD from the native pose, and affinity loss is derived from the mean square error. Then, the best poses are ranked according to score values. 

 ## Model Architecture: Dense and Default2018

**Default2018:** A linear CNN consisting of five convolutional layers
**Dense:** A CNN consisting of twelve convolutional layers. Each layer is organized into three densely connected blocks, following Densenet design principles. Much more accurate, but not as fast as the default2018 model. 

## Repository Structure
| File / Directory | Description |
|------------------|-------------|
| `run_proto_gnina.py` | Main Python pipeline implementing docking, preprocessing, model training, redocking, evaluation, and plotting. |
| `run_gnina_train.py` | Retraining Python script that implements native GNINA docking, preprocessing, and model retraining, for the *full train/val dataset*. |
| `redocking.py` | Redocking Python script that uses the retrained dense and default2018 models to redock the *full test set*. |
| `posebusters_redocking.py` | Redocking Python script that uses the retrained dense and default2018 models to redock the *PoseBusters set*. |
| `evaluation.py` | Python evaluation script for analyzing docking and redocking outputs, and preparing result summaries. |
| `run_gnina.sh` | Shell script that executes the pipeline inside the Docker container. |
| `run_gnina_train.sh` | Shell script that executes the *full dataset* GNINA retraining inside the Docker container. |
| `redocking.sh` | Shell script that redocks the *full test set* using the models inside the Docker container. |
| `posebusters_redocking.sh` | Shell script that redocks the *PoseBusters dataset* using the selected models inside the Docker container. |
| `Dockerfile` | Builds the complete GNINA-Torch environment that supports GNINA-Torch retraining. |
| `gnina_proto_sub.sub` | HTCondor submission script used for running the pipeline on the HPC cluster. |
| `gnina_train.sub` | HTCondor submission script used for running retraining for the *full train/val dataset* on the HPC cluster. |
| `gnina_redock.sub` | HTCondor submission script used for running redocking for the *full test dataset* on the HPC cluster. |
| `gnina_redock_posebusters.sub` | HTCondor submission script used for running *PoseBusters redocking* jobs on the HPC cluster. |
| `proto_plot_inspection.ipynb` | Jupyter notebook for visualization of training metrics and docking results across epochs. |
| `data/proto_train.types` | GNINA-Torch training input dataset generated from docked prototype training dataset poses. |
| `data/full_sealed_train.types` | GNINA-Torch training input dataset generated from docked full training dataset poses. |
| `data/full_sealed_val.types` | GNINA-Torch training input dataset generated from docked validation dataset poses. |
| `Plotting/CNN_metrics_to_csv.py` | Python script for converting GNINA/GNINA-Torch CNN training metrics into CSV format for downstream analysis and plotting. |
| `Plotting/posebusters_plot_inspection.ipynb` | Jupyter notebook for visualizing and inspecting PoseBusters evaluation results, for readability. |
| `results/gninatorch_training/` | Training outputs for each retrained CNN model, including the full PyTorch checkpoint(gnina_retrained_full_model.pt), weight checkpoint after 5 epochs (checkpoint_5.pt), GNINA training logs, and compiled CSV files of the training metrics. |
| `results/plots/` | Images of the plots and metric summaries for the retrained models of the proto data, for readability. |
| `results/proto_test_retrained/` | Docking outputs generated by redocking the test dataset (proto_train) using the retrained CNN models. |
| `results/posebusters_redocked/` | Redocking outputs generated for the PoseBusters dataset using retrained GNINA CNN models. |
| `results/posebusters_filtered_evaluation_default2018_seed1_RMSDonly.csv` | PoseBusters evaluation results for the `default2018` model using RMSD-only metrics. |
| `results/posebusters_filtered_evaluation_default2018_seed1_full.csv` | PoseBusters evaluation results for the `default2018` model using the default evaluation metric set. |
| `results/posebusters_filtered_evaluation_dense_seed1_RMSDonly.csv` | PoseBusters evaluation results for the retrained dense model using RMSD-only metrics. |
| `results/posebusters_filtered_evaluation_dense_seed1_full.csv` | PoseBusters evaluation results for the retrained dense model using the default evaluation metric set. |

## Pipeline Overview

# run_gnina_train.py/.sh:
1) Dock the training (full_data.tar.gz) dataset using the default native GNINA CNN.
2) Generate .types files from the resulting docked training poses.
3) Retrain the two GNINA-Torch CNN models (default2018 and dense) using the generated .types files.
4) Save the full retrained TorchScript models in .pt format.
   
# redocking.py/.sh and posebusters_redocking.py/.sh: 
1) Redock the *full test* and *PoseBusters* dataset using the retrained CNN models.
2) Collect docking metrics, including RMSD, CNN Pose Score, and CNN Affinity.
3) With the plot_inspection.ipynb, generate evaluation plots and training loss plots by parsing the GNINA-Torch training logs and the evaluation.py results. 
   
## Input Data

**The training dataset consists of:**
- receptor structures (*_protein_refined.pdb)
- native ligand structures (*_ligand_refined.sdf)
- binding affinity labels (full_sealed_train/val.csv)

The training data is docked using the original native GNINA model to generate docking poses, which are converted into GNINA-compatible .types training files. The test dataset, unseen by the retrained models, follows the same structure. 

**Preprocessed .types files for retraining:**

GNINA-Torch is trained using .types files, as opposed to explicit protein and ligand structures. Each .types entry contains the following information:
- pose label (good/bad pose, formatted as 1/0, respectively)
- binding affinity (pK)
- receptor structure
- docked ligand pose

## Docker Image
The Docker container is based on the official GNINA Docker container (found at docker.io/gnina/gnina:latest), appending various packages for training and plotting:
1) Python 3.12
2) Miniforge / Conda
3) PyTorch
4) CUDA 12.1
5) PyTorch Ignite
6) LibMolGrid
7) Open Babel
8) MLflow
9) NumPy
10) SciPy
11) Pandas
12) MatPlotLib
13) Scikit-learn

Our Docker image can be found at docker.io/ansc00053/gnina-train:proto-v2. 
Notably, in order to obtain fully functional retrained models, our Docker build patches the original GNINA-Torch training script to export a standalone TorchScript model (named gnina_retrained_full_model.pt) after training to be compatible with GNINA's --cnn_model argument.
## Zenodo Results
Since the checkpoints and the predicted poses of PoseBusters and the test set had big size, these files were uploaded to Zenodo in tar.gz format. 
GNINA Retrained Full Dataset Checkpoints and Redocking Results for PoseBusters Dataset: https://zenodo.org/records/21411180
## Limitations
CNNs are highly accurate, while sacrificing speed. Even at 1 training epoch, our submissions to the cluster lasted up to five hours. With five epochs, training took up to seven hours. This hindered us from making further advancements to our script, such as adding in the knowledge distillation features necessary for training student models. This will be implemented in the final script. Several dependency-related issues stemmed in our Docker image build prevented us from progressing, as well, which proved costly to the time we had. In addition, GNINA-Torch does not accept partial weights-only checkpoint files--due to it only accepting full TorchScript models, we had to append the training.py script that GNINA-Torch uses in order to produce a retrained model that could properly dock the test data. Another limitation that increased our runtime was that GNINA-Torch was not well optimized for multi-GPU runs. Although we submitted jobs with two GPUs, the workflow primarily ran on a single GPU, or had difficulty assigning GPUs.
## Next Steps
1) Implement Kullback-Leibler (KL) divergence loss-based knowledge-distillation to create student models
2) Create GNINA version 1.3 ensembles, using varying seeds 
3) Implement the provided evaluation.py to determine RMSD 
4) Incorporate PoseBusters validation set into training, as well as full training set
