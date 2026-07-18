# Interformer
--------------------
## Benchmarking DL-Based Docking Tools

Team Members: Lakshana Bakthavachalam & Ben Horvath

Supervisors: Hamza Ibrahim & Andrea Volkamer

--------------------
## Final Results
1. Poses predicted by the model: `results/`
1. Full Dataset & Results: `https://zenodo.org/records/21415219`
1. Plots can be found at: `https://wandb.ai/dl-docking/Interformer?nw=nwuserbeho00003` (shared with Hamza)

-----------
## Final Submission Notice: 

Due to memory problems(see file sizes above), we could only run to 5 epochs



-----------
## Notes on File Structure:
1. Renamed full_data/ to data/ to reduce reproducibility steps. Also 
2. Manually separated .pdb and .sdf files from  into:
   - /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train/train_pdb
   - /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train/train_sdf
3. Check README.md files under each steps in condor_workflow/ to understand the workflow of the training
4. Loss curve of energy and affinity normal model - loss_curve/ (Affinity pose not included as trained only for 5 epochs)



--------------------

## Overview of Steps
Please note that more details of each step can be found in the README of each step. 

--------------------
# Interformer Pipeline Summary

| Step | Purpose | Input / Output |
|---|---|---|
| **1: Preprocessing** | Prepares raw protein/ligand data for training: adds hydrogens to ligands (obabel), generates initial 3D conformations (UFF via `rdkit_ETKDG_3d_gen.py`), cleans protein structures (Reduce) to avoid steric clashes, and extracts the binding pocket within 10 Å of the reference ligand (`extract_pocket_by_ligand.py`). Train/val/test sets are each processed with a dedicated script, then combined into one CSV (`combine_csv.py`). | **Input:**<br>Raw folders of `.pdb`/`.sdf` files (full_train/full_val/full_test).<br><br>**Output:**<br>`proto_train`/`proto_val`/`proto_test` folders (separated pdb/sdf, processed pocket, ligand, and UFF-optimized ligand); merged `proto_train_val_final.csv` for training. |
| **2: Model training** | Trains three Interformer model variants using `train.py`: <br> (a) **Energy model** — baseline pose-scoring model; <br>(b) **Affinity normal model** — predicts affinity from original (non-augmented) poses only <br>(c) **Affinity+pose model** — trained on positive/negative pose samples generated via redocking with the energy-model checkpoints (implemented after Step 4). | **Input:**<br>`proto_train_val_final.csv` (and later `proto_train_val_final.round0.csv` for the pose model, after copying redocked `.sdf`s from Step 4 into `data/proto_train/ligand`).<br><br>**Output:**<br>Model checkpoints (`checkpoints/energy_model`, `checkpoints/affinity_normal_model`, `checkpoints/affinity_pose_model`). |
| **3: Gaussian function & complex generation** | Runs inference with the trained energy-model checkpoints to predict Gaussian interaction scores and generate protein-ligand complex structures needed for the subsequent docking/negative-sampling step. | **Input:**<br>`proto_train_val_final.csv` + `checkpoints/energy_model`.<br><br>**Output:**<br>Gaussian interaction predictions and generated complex structures (feeds into Step 4's docking folder). |
| **4: Docking poses generation** | Reconstructs ligand poses from the Step 3 outputs, generates docking statistics (RMSD, pose rank, van der Waals distance, energy), and merges these stats back into the training CSV. Poses with RMSD ≤ cutoff (default 2 Å) are kept as positive samples poses above the cutoff are relabeled with negative pIC50 via `label_negatives.py` to create contrastive positive/negative training data for the affinity+pose model. | **Input:**<br>Complex/Gaussian/ligand/UFF outputs from Step 3 (unpacked from tarballs).<br><br>**Output:**<br>Redocked `.sdf`'s (`ligand_reconstructing`), `stat_concated.csv`, and the labeled `proto_train_val_final.round0.csv` (with signed pIC50 for pos/neg samples). |
| **5: Final docking pose prediction (test set)** | Repeats Steps 3–4 on the held-out test set, then runs final inference using the trained affinity+pose model checkpoints to score docking poses and predict pIC50/pose scores for the test ligands. | **Input:**<br>`proto_test_final.round0.csv` + affinity+pose model checkpoints (redocked test `.sdf`s copied into `data/proto_test/ligand`).<br><br>**Output:**<br>Final scored CSV `proto_test_final.round0_ensemble.csv` (predIC50, pred_pose_score); <br>top-1 pose used downstream in `evaluation.py`. |

---

## Repository Directory Layout

| Folder | Description |
|:---|:---|
| `checkpoints/` | Contains saved model weights across variants (Energy, Affinity normal, and Affinity+pose models). |
| `condor_workflow/` | HTCondor cluster execution scripts and subdirectories. Includes dedicated README files per step to track the orchestration workflow. |
| `data/` | Root dataset folder (renamed from `full_data/`). Contains split subdirectories (`proto_train/`, `proto_val/`, `proto_test/`) for clean protein/ligand structures and redocked assets. |
| `docking/` | Directory for managing docking run configurations, unpacking intermediate tarballs, or staging contrastive analysis. |
| `eda/` | Exploratory Data Analysis notebooks or scripts analyzing dataset properties and training characteristics. |
| `images/` | General imagery, workflow diagrams, or asset files used for documentation. |
| `interformer/` | Core package source code, neural network architecture scripts, and module logic for the Interformer models. |
| `lightning_logs/` | PyTorch Lightning runtime logs, tracking training progress checkpoints, configurations, and internal telemetry metrics. |
| `loss_curve/` | Storage folder for generated plots tracking energy and affinity normal model loss trajectories. |
| `results/` | Output repository for model evaluations, including final predicted test set poses and ensemble scored CSV metrics. |
| `scripts/` | Helper files and pipeline execution scripts (e.g., UFF 3D generation, binding pocket extractions, data mergers, or negative labeling). |
| `tools/` | External utilities, binary tools (e.g., OpenBabel/Reduce integrations), or data harvesting code assets. |
---


## Script Execution & Pipeline Utilities

### Pipeline Phase Legend
* **(1)** Data Preparation & Cleaning
* **(2)** Feature & Annotation Extraction
* **(3)** Model Training & Inference
* **(4)** Docking Reconstruction & Post-Processing

<br>

| File Path / Folder Structure | Description |
|:---|:---|
| (1)<br>`create_target_train.py`<br>`create_target_test.py` | Builds a standardized "Target" identifier column (`PDBID_Ligand_Chain_Residue` for train/val; parsed from the ligand filename for test) and inserts it as the first column so downstream Interformer scripts can key on a single complex ID. |
| (1)<br>`rename_docked_sdf_train.py`<br>`rename_docked_sdf_test.py` | Batch-renames OpenBabel output ligand files from the `*_ligand_refined.sdf` suffix to the `*_docked.sdf` suffix required by Interformer's downstream docking/pipeline scripts. |
| (1)<br>`combine_csv.py` | Concatenates the finalized prototype train and validation CSVs into a single combined train + val CSV for full-dataset model training. |
| (2)<br>`get_IC50_uniprot_train.py`<br>`get_IC50_uniprot_test.py` | Computes pIC50 (-log Binding Affinity) for each entry in the prototype train/test CSVs and cross-references PDB IDs against the RCSB GraphQL API (batched) to append missing UniProt IDs, replicating a feature present in the original dataset but absent from the prototype data. |
| (2)<br>`rdkit_ETKDG_3d_gen.py` | Generates ligand conformers using RDKit's ETKDGv3 algorithm (30 embeddings), UFF-optimizes each, and saves the lowest-energy conformer as the `_uff.sdf` reference structure, reporting energy and RMSD vs. the original input. |
| (2)<br>`extract_pocket_by_ligand.py` | Extracts the protein binding pocket (10 Å cutoff) around a reference ligand using ODDT's `ExtractPocketAndLigand`, identifying and optionally excluding the ligand's own CCD code so other bound heteroatoms (metals, cofactors) are preserved in the output pocket PDB. |
| (3)<br>`train.py` | Main PyTorch Lightning training entry point for Interformer (energy, pose, or affinity-normal models); configures the data module, model, and Weights & Biases logging. Possible to run distributed (DDP- Distributed Data Parallel) training via the Trainer. |
| (3)<br>`inference.py` | Inference across an ensemble of trained Interformer checkpoints to predict pIC50/pose scores per target, then selects the top-8 performing models and blends their outputs into a consensus ensemble CSV with prediction variance and Pearson correlation metrics. DDP runs are possible for faster training.|
| (4)<br>`reconstruct_ligands.py` | CLI driven docking reconstruction tool that runs Monte Carlo pose sampling (64 repeats x 2000 steps) per target using predicted energy grids—either locally or via remote/cluster dispatch. Generates docked ligand poses, compiles a detailed docking summary (RMSD, energy, torsions, pose rank), and aggregates per-target results into a summary CSV reporting RMSD pass rate and steric clash pass rate across all docked targets. <br>
| (4)<br>`merge_summary_input.py` | Merges the docking summary statistics (`pose_rank`, `num_torsions`, `energy`, `rmsd`) produced by the `reconstruct_ligands.py` pipeline back into the main input datasets. |

---

## Conclusion

We have learned a lot from this experience to experiment and reproduce results for Interformer and would like to thank Revo Lai(an Interformer author) and Hamza for the mentorship; as well as Prof. Volkamer the opportunity to partake in this seminar!
