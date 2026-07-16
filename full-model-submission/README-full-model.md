# DLDockingBenchSeminar — Uni-Mol Docking V2 (Full Dataset Submission)

**Cluster Username:** bdldt\_team007  
**Team Members:** Sadaf Reihani, Elnaz Abdollahzadeh  
**Email Addresses:** sare00008@stud.uni-saarland.de , elab00003@stud.uni-saarland.de  
**Base model:** [Uni-Mol Docking V2](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol_docking_v2)  
**Cluster:** HTCondor on conduit.hpc.uni-saarland.de  
**Docker image:** docker.io/elab00003/unimol:1

**Zendo link:**  https://zenodo.org/records/21300137?preview=1\&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc3OGJlOWI2LWY2NDEtNGUxYy1hNmM0LThiNjcwZjdkOWU2MSIsImRhdGEiOnt9LCJyYW5kb20iOiIwODk0YTY5OWY4YjQ1NTJkZjU2ZDAyMGMzNjNkOGEyZiJ9.8H9WB7e8RVf3OREl5ZdQGUo2lYRDGananzC7KnolVE\_CfF4FOF8HXw4lUUTnPRWwUiOWt-SOwTKQni6lqPqeAQ

\---

## What this submission does

We fine-tune Uni-Mol Docking V2 on the **full LP-HiQBind dataset** (23,447 training complexes) and evaluate predicted binding poses on the validation set and test set. We ran 7 training experiments with different hyperparameters to find the best configuration. As a bonus, we also evaluate on the PoseBusters benchmark.

The full pipeline:

```
raw .sdf/.pdb files
      │
      ▼  prepare\\\_data.py
full\\\_train.lmdb / full\\\_valid.lmdb
      │
      ▼  train\\\_run6.sh  (fine-tuning, best run)
checkpoint\\\_best.pt
      │
      ▼  infer\\\_run6.sh
valid.pkl / test.pkl  (raw predictions)
      │
      ▼  pkl\\\_to\\\_sdf\\\_run6.py
{complex\\\_id}\\\_pred.sdf  (one file per complex)
      │
      ▼  evaluation.py
RMSD results
```

\---

## Dataset

**Source:** LP-HiQBind dataset, available at https://zenodo.org/records/20764848

|Split|Complexes|Used for|
|-|-|-|
|full\_train|23,447|Training all 7 runs|
|full\_valid|2,609|Validation during training|
|full\_test|6,183 (6,181 after cleaning)|Final test evaluation|

\---

## Hyperparameter Search

We ran 7 training experiments to find the best hyperparameters:

|Run|LR|Recycling|Batch|fp16|Epochs|Notes|
|-|-|-|-|-|-|-|
|main\_run|1e-5|1|8|No|30|Conservative baseline|
|second\_run|1e-5|3|8|Yes|15|Testing more recycling|
|run3|3e-5|1|8|Yes|10|Higher LR|
|run4|1e-5|4|8|Yes|10|Paper's recycling|
|run5|3e-5|4|8|Yes|10|Higher LR + paper recycling|
|**run6 (BEST)**|**3e-4**|**1**|**32**|**Yes**|**30**|**Paper's LR + larger batch**|
|run7|3e-4|4|16|Yes|10|Paper's LR + recycling|

**Why Run 6 won:** The original paper trains with lr=3e-4 and batch=64 (across 8 GPUs). Our earlier runs used lr=1e-5 which was too conservative — the model learned too slowly given our compute budget. Matching the paper's learning rate with batch=32 gave the largest single improvement, dropping median RMSD from \~5.37 Å (prototype) to 3.73 Å.

**Hardware:** A100 GPU (40GB VRAM), granted for full dataset training.

\---

## Results

### Main Results (Run 6 — best model)

|Dataset|Median RMSD|RMSD < 2.0 Å|RMSD < 1.0 Å|
|-|-|-|-|
|Prototype baseline|5.37 Å|2.2%|0.0%|
|Validation set (epoch 10)|3.73 Å|8.3%|0.9%|
|Validation set (epoch 30)|3.79 Å|\~8%|\~0.9%|
|Test set — Run 6|3.97 Å|9.2%|0.9%|
|Test set — Run 7|3.87 Å|7.5%|0.8%|

\---

## 

## Bugs Fixed

### 

### 1\. `--reset-optimizer` flags prevented proper checkpoint resumption

The original training script had these flags:

```
--reset-optimizer
--reset-dataloader
--reset-meters
--reset-lr-scheduler
```

On HTCondor, training jobs are regularly preempted (evicted) and must resume from checkpoints. These flags caused training to restart the optimizer state from scratch every time, effectively restarting training. Removing them enabled proper checkpoint resumption — the optimizer momentum, learning rate schedule, and data position are all restored correctly.

### 

### 2\. Conformer count mismatch crashes inference (`IndexError`)

`TTADockingPoseDataset` crashes with `IndexError: list index out of range` if any LMDB entry has fewer conformers than `--conf-size`. This happens because RDKit occasionally fails to generate the full requested number of conformers for certain molecules.

We discovered one entry in the test set (`6cvw\\\_FH1\\\_A\\\_1206`, 7 conformers instead of 10) caused this crash. Since `TTADockingPoseDataset` is frozen prototype code we cannot modify, we wrote `check\\\_conf\\\_distribution.py` to detect and drop such entries before inference, creating `test\\\_clean.lmdb`.

This must be run on any new test data before inference:

```bash
python3 check\\\_conf\\\_distribution.py data/processed\\\_test/test.lmdb 10 \\\\
    --drop data/processed\\\_test/test\\\_clean.lmdb
```

### 

### 3\. `--warmup-ratio` crashes on checkpoint resume (unicore bug)

The original script used `--warmup-ratio 0.06` which requires computing `total\\\_train\\\_steps = n\\\_samples / batch\\\_size × epochs` at startup. This computation fails when resuming from a checkpoint because the dataloader state isn't fully initialized yet — a bug in the unicore framework.

**Fix:** replace with explicit fixed values computed manually:

```
--warmup-updates 1319        # = 6% of total steps for run6 (batch=32, 30 epochs)
--total-num-update 21990     # = 733 steps/epoch × 30 epochs
```

### 

### 4\. `--max-update 10000` stopped training too early

An early training script included `--max-update 10000` as a safety cap. With batch=32 and 23,447 training examples, one epoch = \~733 steps. The cap of 10,000 updates would stop training at \~epoch 14, far short of the intended 30 epochs.

**Fix:** removed `--max-update` entirely, matching the original paper which uses only `--max-epoch` to control training length.

\---

## 

## Large Files (Zenodo)

Checkpoints, predicted poses, and full training LMDB are too large for GitHub and are stored on Zenodo:

**https://zenodo.org/record/21300137**

|File|Description|
|-|-|
|`run6\\\_checkpoint\\\_best.pt`|**Best model** — Run 6, epoch 30, lr=3e-4, batch=32|
|`run7\\\_checkpoint\\\_best.pt`|Run 7 best checkpoint — lr=3e-4, recycling=4|
|`run3\\\_checkpoint\\\_best.pt`|Run 3 best checkpoint — lr=3e-5, recycling=1|
|`run4\\\_checkpoint\\\_best.pt`|Run 4 best checkpoint — lr=1e-5, recycling=4|
|`run5\\\_checkpoint\\\_best.pt`|Run 5 best checkpoint — lr=3e-5, recycling=4|
|`main\\\_run\\\_checkpoint\\\_best.pt`|Main run best checkpoint — lr=1e-5, fp32|
|`full\\\_train.lmdb`|Preprocessed training LMDB (228 MB)|
|`run6\\\_valid\\\_poses.zip`|Predicted validation poses — 2,408 SDF files|
|`run6\\\_test\\\_poses.zip`|Predicted test poses — 5,298 SDF files (Run 6)|
|`run7\\\_test\\\_poses.zip`|Predicted test poses — 5,298 SDF files (Run 7)|
|`full\\\_test.zip`|Raw test dataset (SDF + PDB files)|

\---

## Tools and Resources Used

* **Cluster:** `conduit.hpc.uni-saarland.de`, HTCondor scheduler
* **Account:** `bdldt\\\_team007` — A100 GPU (40GB VRAM) access granted for full dataset training
* **Container:** `docker.io/elab00003/unimol:1` (PyTorch 1.12, RDKit, LMDB, UniCore)





\---

## Files in This Submission

|Folder|File|Description|
|-|-|-|
|`data/preprocessing/`|`prepare\\\_data.py`|Raw SDF/PDB → LMDB (unchanged from prototype)|
||`prepare\\\_data\\\_full.py`|Full dataset preprocessing wrapper|
||`prepare\\\_test.py`|Test set preprocessing|
||`check\\\_conf\\\_distribution.py`|Detects + drops entries with too few conformers (see Bug #2)|
||`condor\\\_\\\*.sub`|HTCondor job files for preprocessing|
|`training/main\\\_run/`|`train\\\_full.sh`|main\_run training script (lr=1e-5, batch=8, fp32)|
|`training/run3/`|`train\\\_run3.sh`|run3 training script (lr=3e-5, recycling=1)|
|`training/run4/`|`train\\\_run4.sh`|run4 training script (lr=1e-5, recycling=4)|
|`training/run5/`|`train\\\_run5.sh`|run5 training script (lr=3e-5, recycling=4)|
|`training/run6/`|`train\\\_run6.sh`|**Best run** training script (lr=3e-4, batch=32)|
|`training/run7/`|`train\\\_run7.sh`|run7 training script (lr=3e-4, recycling=4, batch=16)|
|`training/second\\\_run/`|`train\\\_exp2.sh`|second\_run training script (lr=1e-5, recycling=3)|
|`inference/`|`infer\\\_run6.sh`|Run 6 inference on validation set|
||`infer\\\_test\\\_run6.sh`|Run 6 inference on test set|
||`infer\\\_test\\\_run7.sh`|Run 7 inference on test set|
||`condor\\\_\\\*.sub`|HTCondor job files for inference|
|`postprocessing/`|`pkl\\\_to\\\_sdf.py`|Original prototype pkl→SDF converter|
||`pkl\\\_to\\\_sdf\\\_run6.py`|Run 6 validation poses|
||`pkl\\\_to\\\_sdf\\\_test.py`|Run 6 test poses|
||`pkl\\\_to\\\_sdf\\\_test\\\_run7.py`|Run 7 test poses|
|`evaluation/`|`evaluation.py`|RMSD evaluation (unchanged from prototype)|
||`convert\\\_and\\\_eval.sh`|Converts full\_val.csv format + runs evaluation|
||`logs/full\\\_val\\\_evaluation.csv`|Run 6 validation results|
||`logs/full\\\_test\\\_evaluation.csv`|Run 6 test results|
||`logs/full\\\_test\\\_evaluation\\\_run7.csv`|Run 7 test results|
|`data/lmdbs/`|`full\\\_valid.lmdb`|Preprocessed validation LMDB|
||`full\\\_test.lmdb`|Preprocessed test LMDB (original)|
||`full\\\_test\\\_clean.lmdb`|Cleaned test LMDB (1 bad entry dropped)|
||`full\\\_train.lmdb`|On Zenodo (228 MB)|
|`utils/`|`inspect\\\_lmdb.py`|Inspect any LMDB file|
||`inspect\\\_pkl.py`|Inspect inference pkl output|





## PoseBusters Evaluation

### 

### PoseBusters Results (Run 6)

||Median RMSD|RMSD < 2Å|PB-Valid|
|-|-|-|-|
|Raw model predictions|3.55 Å|8.1%|0.0%|

### 

### Why PB-Valid = 0% on raw predictions

This is a known, documented limitation of direct coordinate prediction methods — not a training bug. The model predicts atom positions (X,Y,Z) without enforcing chemical bond geometry. When `pkl\\\_to\\\_sdf.py` overwrites atom positions with model predictions, bonded atom distances become physically unrealistic (e.g. a C-C bond that should be 1.5 Å might become 5 Å). PoseBusters correctly flags these as invalid.



This exact issue is documented by Buttenschoen et al. (2024) for all tested deep learning docking methods, including the original Uni-Mol. Even the original fully trained Uni-Mol achieves only \~2% on the combined RMSD+PB-Valid metric without post-processing (Re-Dock, arXiv:2402.11459).

### 

### PoseBusters preprocessing — custom script required

`prepare\\\_data.py`'s `make\\\_lmdb()` cannot process PoseBusters data directly because:

1. **Different CSV format:** `make\\\_lmdb()` expects a `PDBID` column and `\\\_ligand\\\_refined.sdf` filename suffix. PoseBusters files use `5SAK\\\_ZRY\\\_ligand.sdf` format — no PDBID column, no `\\\_refined` suffix.
2. 
3. **Explicit hydrogens at 0.0 Å:** PoseBusters SDF files contain explicit hydrogens at 0.0 Å positions, causing `check\\\_coords()` to reject all 308 complexes.
4. 

**Fix:** `run\\\_prepare\\\_posebusters.sh` — imports core functions from `prepare\\\_data.py` unchanged, but handles the different file format and hydrogen issue.



### Why `infer.py` instead of `demo.py` for PoseBusters

The Uni-Mol repository provides `interface/demo.py` which uses `--mode batch\\\_one2one --steric-clash-fix`. We used `infer.py` instead because `demo.py` requires a completely different input format (JSON docking grid files, separate protein/ligand folders, specific CSV format with `pdb\\\_code`/`lig\\\_code` columns) that is incompatible with our LMDB-based pipeline.

### 

### PoseBusters-related files (Zenodo)

|File|Description|
|-|-|
|`run6\\\_posebusters\\\_poses.zip`|Predicted PoseBusters poses — 308 SDF files (raw)|
|`posebusters\\\_filtered.zip`|Raw PoseBusters dataset|

### 





















