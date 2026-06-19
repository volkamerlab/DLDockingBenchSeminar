# DLDockingBenchSeminar —  Uni-Mol Docking V2

**cluster Username:** bdldt_team007
**Team Members:** Sadaf Reihani, Elnaz Abdollahzadeh
**Email Addresses:**sare00008@stud.uni-saarland.de , elab00003@stud.uni-saarland.de
**Base model:** [Uni-Mol Docking V2](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol_docking_v2)
**Cluster:** HTCondor on conduit.hpc.uni-saarland.de
**Docker image:** docker.io/elab00003/unimol:1

---

## What this project does

We fine-tune Uni-Mol Docking V2 on a small subset of the protein-ligand
binding dataset (712 training complexes, 136 test complexes) and evaluate
how well the resulting model predicts 3D binding poses.

The full pipeline:

```
raw .sdf/.pdb files
      │
      ▼  prepare_data.py
train.lmdb / valid.lmdb
      │
      ▼  train.sh (fine-tuning)
checkpoint_best.pt
      │
      ▼  infer.sh
valid.pkl (raw predictions)
      │
      ▼  pkl_to_sdf.py
{complex_id}_pred.sdf  (one file per complex)
      │
      ▼  evaluation.py
RMSD results
```

---

## Errors we hit and how we debugged/fixed them

### 1. Training crashed instantly: FloatingPointError: Gradients are Nan/Inf

This happened on literally the first training batch, every single time,
no matter what learning rate, batch size, or weight initialization we used.

**Debugging steps we went through:**
- Checked GPU hardware (nvidia-smi, ECC errors) — ruled out, nothing wrong
- Wrote a script to scan every LMDB entry for NaN/Inf coordinates — found nothing, data was clean
- Tried multiple learning rates (1e-3, 1e-4, 1e-6) — same crash every time
- Tried random weight initialization instead of pretrained weights — same crash
- Added debug print statements inside the model (around holo_distance_project) to trace exactly where things went wrong

**What we found:** `mol_pair_decoder_rep` legitimately contains `-inf`
values at padding positions (normal behavior — that's just how attention
masking works). The original code concatenated this straight into
`holo_encoder_pair_rep` without cleaning it up first. A few lines later,
inside `TransformerEncoderWithPair`, a line computes
`attn_mask - input_attn_mask`. When both sides have `-inf` at the same
padding spot, that's `(-inf) - (-inf) = NaN`, and that NaN wrecks every
gradient during backprop.

**The fix** (one line, in `unimol/models/docking_pose_v2.py`, placed
**before** the `torch.cat` that builds `holo_encoder_pair_rep`):

```python
mol_pair_decoder_rep = torch.nan_to_num(mol_pair_decoder_rep, nan=0.0, posinf=0.0, neginf=0.0)
```

We first tried putting this line *after* the `torch.cat` by mistake — it
did nothing, since `holo_encoder_pair_rep` had already copied the bad
values in by then. Moving it before the `torch.cat` fixed it completely.

This is a bug in the original released Uni-Mol Docking V2 code, not in
our data or hardware. After the fix, training ran cleanly for the full 30 epochs.


### 2. Pocket atoms were stored as raw PDB names instead of element symbols

Early on, `pocket_atoms` in our LMDB held names like `CA`, `OD1`, `NE2`
(real PDB atom names) instead of plain element symbols (`C`, `N`, `O`).
The model's vocabulary only knows element symbols, so every pocket atom
was coming through as `[UNK]`. This was a contributing factor to the
NaN issue early on. We added a `pdb_name_to_element()` helper in
`prepare_data.py` that converts PDB atom names to their correct element
symbol before storage.

---

### 3. Duplicate PDBIDs caused wrong output filenames during postprocessing

We initially stored just the bare PDBID (e.g. `3ix2`) in the LMDB's
`pocket` field. The problem: a single PDB structure can contain several
different ligand copies (e.g. `3ix2_AC2_A_302`, `3ix2_AC2_B_302`,
`3ix2_AC2_C_302` all share the PDBID `3ix2`). This made it impossible to
tell predictions for different complexes apart during postprocessing.

**Fix:** derive a proper unique `complex_id` from the ligand filename
instead, by stripping the `_ligand_refined.sdf` suffix:

```python
complex_id = row['ligand_file_name'].replace('_ligand_refined.sdf', '')
```

We verified this fix worked by checking that all 136 test entries now had
136 unique pocket/complex IDs (previously many shared the same ID).

---

### 4. Postprocessing: atom count mismatch when rebuilding molecules

The official Processor class (used for single-molecule prediction)
expects a mol_list key pre-stored in the LMDB with RDKit molecule
objects already inside. Our LMDB doesn't have that — it only stores a
smi (SMILES) field. So we wrote our own pkl_to_sdf.py that rebuilds
the molecule from SMILES instead.

First attempt failed for every complex with `atom count mismatch
(coords=32, mol=65)` — the predicted coordinates only cover heavy atoms
(hydrogens get filtered out by the model's `token_mask`), but we were
trying to set those coordinates on a molecule that still had explicit
hydrogens attached. Fix: call `Chem.RemoveHs(mol)` right after parsing
the SMILES, before doing anything else with the molecule.

Result: 135 out of 136 SDF files written successfully. The one remaining
failure (`5ree_T1M_A_404`) is RDKit failing to generate any 3D conformer
for that particular SMILES — a known occasional RDKit limitation, not a
bug in our code.

---


## Verifying pipeline correctness


 pipeline correctness using the original pretrained
checkpoint.** Our own fine-tuned model's predicted poses all came out
with every atom collapsed to nearly the same point (see `debug_coords.py`
for the raw numbers). To check whether this was a bug in our inference/
postprocessing code or just an undertrained model, we ran the exact same
`infer.py` → `pkl_to_sdf.py` pipeline using the **original pretrained
Uni-Mol Docking V2 checkpoint** (`weights/run0_pose_new_PDBbind_pose_
recycling_4_lr_0.0003_bs_32_dist_th_8.0_epoch_200_wp_0.06/checkpoint_best.pt`,
465MB, trained on the full PDBbind dataset for 200 epochs). Result:
properly spread-out, physically reasonable poses. This confirmed our
pipeline code is correct, and the collapsed poses from our own checkpoint
are a training-scale problem, not a software bug. See
`pkl_to_sdf_sanity.py`.

**Conclusion:** preprocessing, inference, and postprocessing are all
verified correct. Our undertrained model (712 samples, 30 epochs,
batch_size=8, recycling=1, single P100) collapses to a degenerate
solution instead of learning real 3D geometry — a known failure mode for
coordinate-regression models that haven't seen enough data/training.

---

## Tools and resources used

- **Cluster:** `conduit.hpc.uni-saarland.de`, HTCondor scheduler
- **Account:** `bdldt_team007` — P100 GPUs
- **Container:** `docker.io/elab00003/unimol:1` (PyTorch 1.12, RDKit, LMDB, UniCore)
- **Chemistry tools:** RDKit (conformer generation, SMILES parsing,
  SDF writing), biopandas (PDB parsing)

---

## Files in this repo

| File | What it does |
|---|---|
| `prepare_data.py` | Raw files → train.lmdb / valid.lmdb |
| `train.sh`, `run_train.sh`, `condor_train.sub` | Fine-tuning |
| `infer.sh`, `run_infer.sh`, `condor_infer.sub` | Run model on test set → valid.pkl |
| `pkl_to_sdf.py`, `run_pkl_to_sdf.sh`, `condor_pkl_to_sdf.sub` | valid.pkl → individual .sdf files |
| `evaluation.py` | RMSD evaluation |
| `unimol/models/docking_pose_v2.py` | Model architecture — **contains our fix** |
| `unimol/tasks/docking_pose_v2.py` | Data loading (unmodified) |
| `unimol/losses/docking_pose_v2.py` | Loss computation (unmodified) |
| `results_notebook.ipynb` | Loss curves, RMSD plots, comparison to published results |
| `inspect_pkl.py`, `inspect_lmdb.py` | Debugging: checked data structure |
| `debug_coords.py`, `pkl_to_sdf_sanity.py` | Debugging: traced the collapsed-pose issue |

---

## Results

| Metric | Value |
|---|---|
| Training samples | 712 |
| Test samples evaluated | 135 / 136 |
| Best validation loss | 8.256 |
| Final validation RMSD | 6.537 Å |
| RMSD < 2.0 Å | 2.2% |
| RMSD < 1.0 Å | 0.0% |
| Median RMSD | 5.37 Å |

The paper reports 77.6% RMSD < 2Å on PoseBusters (N=428), trained on the
full MOAD dataset with 8× V100 GPUs, batch_size=64, 100 epochs,
recycling=4. Our much smaller scale (712 samples, 1× P100, 30 epochs,
batch_size=8, recycling=1) explains the gap confirmed by our sanity
checks above to be a training-scale issue, not a pipeline bug.
