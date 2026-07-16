# KarmaDock — Prototype Submission

**Benchmarking DL-based Docking Tools** · Saarland University, Summer 2026
supervisors: Hamza Ibrahim, Andrea Volkamer · **Team (KarmaDock): Ahmed, Abdullah**

Docking = given a protein pocket and a small molecule (ligand), predict the ligand's bound
3D **pose**. KarmaDock is a deep-learning docker: it builds protein+ligand graphs, encodes
them (GVP + graph-transformer), **moves** the ligand into the pocket with an E(n)-equivariant
GNN (recycled ×3), and **scores** the pose with a mixture-density network (MDN). It generates
the pose directly instead of searching, so it is ~100–1000× faster than classical docking.
Paper: Zhang et al., *Nat. Comput. Sci.* **3**, 789–804 (2023),
[doi:10.1038/s43588-023-00511-5](https://doi.org/10.1038/s43588-023-00511-5).

---

## 🚧 Final submission (full-data) — report in progress

The tables below use the prototype `proto_test` (136). The **final full-data submission** is being
finalized: our from-scratch model retrained on the full seminar split
(`model/full_scratch_karmadock_team002.pkl`; 2-stage paper protocol, Stage-2 via 2×A100 (40GB each); evaluated head-to-head against the authors' released weights on the
**same** `full_test` (6,183) and `posebusters_filtered` (308) sets.

**Preliminary headline** (top-1 success@2 Å, using `evaluation.py`, uncorrected):

| set | ours (full-data) | released weights |
|---|---|---|
| full_test (6,183) | 82.2 % | 88.3 % |
| PoseBusters (308) | 76.9 % (PB-Valid 5.2 %) | 83.1 % (PB-Valid 2.6 %) |

This commit adds the trained model (`model/`), the evaluation input CSVs (`data/`), the HPC condor
submit files (`condor/full_stage2_2gpu.sub` = the 2×A100 Stage-2 run;
`condor/{full_test,posebusters}_infer.sub` = inference) and their job logs (`condor/logs/`). The
full report and results notebook are being finalized.

**Predicted poses** (all 3 variants, both datasets, both models — too large for git) are on Zenodo:
[zenodo.org/records/21197043](https://zenodo.org/records/21197043).

---

**Contents:**
1. [What we did & why](#1-what-we-did-and-why-changes-vs-upstream-karmadock)
2. [The three pipelines](#2-the-three-pipelines) — incl. [workflow diagrams](#workflow)
3. [Results](#3-results)
4. [How to evaluate / reproduce](#4-evaluate--reproduce)
5. [Training parameters](#5-training-information--parameters-from-the-paper)
6. [Repository layout](#6-repository-layout)
7. [Issues & fixes](#7-issues--fixes)
---

## 1. What we did, and why (changes vs. upstream KarmaDock)

Public KarmaDock ships **inference + released weights only — there is no training script.**
Our contribution is everything needed to *train* it and to *benchmark* it reproducibly on the
seminar split. The upstream **model, preprocessing, and docking code are used UNMODIFIED** —
this keeps the benchmark faithful to the published method.

| File (ours) | What it is | Why it exists |
|---|---|---|
| [**`scripts/train.py`**](scripts/train.py) | full checkpointed training loop (MDN + docking RMSD losses, 2-stage ) | **our main artifact** — upstream has no trainer; required to train from scratch / fine-tune |
| [`scripts/run_train.sh <scratch/finetune>`](scripts/run_train.sh) | preprocess + train: `scratch` = paper's 2-stage protocol (**P2**), `finetune` = single-stage from the released weights (**P3**) | reproduce the paper's *Training protocol* (Methods, p.801) |
| [`scripts/convert_seminar_to_karmadock.py`](scripts/convert_seminar_to_karmadock.py) | seminar data layout → KarmaDock layout | the seminar's `<id>_ligand_refined.sdf` / `<id>_protein_refined.pdb` files must be re-laid-out for KarmaDock's `pre_processing.py` |
| [`scripts/convert_karmadock_to_seminar.py`](scripts/convert_karmadock_to_seminar.py) | KarmaDock poses → seminar `results/<ds>/<id>_pred.sdf` (best-pose-first) | produce the exact format `evaluation.py` expects |
| [`scripts/run_infer.sh`](scripts/run_infer.sh) | preprocess → dock → export the **3 pose variants** (uncorrected / FF / align) | produce the predicted poses on the cluster |
| [`scripts/evaluate.sh`](scripts/evaluate.sh) | run `evaluation.py` over every pipeline × variant — the separate scoring step | produce the official RMSD CSVs |
| [`condor/`](condor)`*.sub` | HTCondor docker-universe submit files (3 docking jobs + 1 eval job + 2 training jobs) | run everything on the SIC cluster |
| [`Dockerfile`](Dockerfile) | image → `ahlamloum/karmadock-seminar:v6` | reproducible environment (KarmaDock + RDKit + Torch + other dependencies backed in) |

## 2. The three pipelines

| pipeline | what it is | weights |
|---|---|---|
| **P1 — baseline** | released `karmadock_screening.pkl`, inference only | upstream (in image) |
| **P2 — from scratch** | paper's 2-stage protocol, trained on `proto_train` (712) | [`model/p2_scratch_karmadock_team002.pkl`](model/p2_scratch_karmadock_team002.pkl) |
| **P3 — fine-tune** *(bonus)* | released weights fine-tuned on `proto_train` (712) | [`model/p3_finetune_karmadock_team002.pkl`](model/p3_finetune_karmadock_team002.pkl) |

*Fine-tuning (P3) is a bonus, not a core requirement and still under development.*

### Workflow

**① Training — produces the P2 / P3 checkpoints:**

<a href="docs/workflow_training.png"><img src="docs/workflow_training.png" alt="Training workflow" width="580"></a>

*Preprocess `proto_train` (712) into graphs, then train: `--init_model` picks the route — **P2** from scratch (Stage 1 MDN scoring → Stage 2 + docking RMSD) or **P3** fine-tune from the released weights; `Early_stopper` keeps the best epoch as the checkpoint.*

**② Inference & evaluation:**

<a href="docs/workflow_inference.png"><img src="docs/workflow_inference.png" alt="Inference and evaluation workflow" width="700"></a>

*Run once per pipeline (P1 / P2 / P3): preprocess `proto_test` (136), dock + score with the chosen weights, export the 3 pose variants (uncorrected / FF / align), then score with the official `evaluation.py` (symmetry-corrected RMSD, top-1).*



## 3. Results

Our submission model is the **from-scratch P2** (the seminar task — retrain the tool on the
shared split); P1 (released baseline) and P3 (fine-tune, bonus) are shown for context.
Scored by the official [`evaluation/evaluation.py`](evaluation/evaluation.py)  on the **136-complex `proto_test`**, for all three KarmaDock pose
post-processing variants.

**success@2 Å (top-1):**

| pipeline | uncorrected | FF-corrected | align-corrected |
|---|---|---|---|
| **P2 — from scratch** (our model) | **10.3 %** | 11.0 % | 94.1 % |
| P1 — baseline (released) | 80.9 % | 78.7 % | 95.6 % |
| P3 — fine-tune *(bonus)* | 80.1 % | 75.0 % | 94.9 % |

Uncorrected **@1 Å / median RMSD**: P2 3.7 % / 3.38 Å · P1 8.1 % / 1.45 Å · P3 7.4 % / 1.48 Å.
Per-complex CSVs are in [`results/`](results) (`<pipeline>_<variant>_evaluation.csv`, and
`proto_test_evaluation.csv` = the P2 headline). Numbers are deterministic (`--random_seed 2023`)

> we belive that align-corrected is misleading as according to the paper it should be lower than the uncorrected and FF-corrected results which is the opposite of what we got. [ need futher investigation - doesn't affect the training process for the next phase ]

The **uncorrected** pose is the
raw model output. The **FF** variant is a force-field relaxation. The **align-corrected** variant
superimposes the predicted ligand onto the reference frame.

## 4. evaluate / reproduce

Both the predicted poses (shipped **unzipped**) and
the official [`evaluation.py`](evaluation/evaluation.py) result CSVs are included — read the results directly, or re-run the
evaluator on the shipped poses (if needed - locally no cluster needed).

### A. Score the shipped poses
```bash
# one-time setup: unzip the reference structures and let evaluation.py find them
unzip -o data/prototype_model_data.zip -d data/prototype_model_data
ln -sf prototype_model_data/proto_test data/proto_test      # data/proto_test.csv already ships

# score results/proto_test/ (= our from-scratch P2 model) -> results/proto_test_evaluation.csv
python evaluation/evaluation.py --dataset proto_test
```
`results/proto_test/` holds the **from-scratch (P2)** poses — our submission's primary model. To
score a different pipeline/variant, point `results/proto_test/` at it, e.g.
`rm results/proto_test && ln -s p1_baseline/proto_test results/proto_test` then re-run. Pre-computed
CSVs for every pipeline × variant are in [`results/`](results).

### B. Regenerate the poses on the cluster (optional)
```bash
mkdir -p logs
# dock: each job preprocesses, docks (seed 2023) and exports the 3 pose variants
condor_submit condor/p1_baseline.sub        # -> results/p1_baseline/{proto_test,_ff,_align}
condor_submit condor/p2_scratch_infer.sub   # -> results/p2_scratch/...
condor_submit condor/p3_finetune_infer.sub  # -> results/p3_finetune/...
# score (after the 3 docking jobs finish): runs evaluation.py over every pipeline x variant
condor_submit condor/evaluate.sub
```
The subs ([`condor/`](condor)) are **portable** (they transfer the data + our code into the job and use the
released weights baked in the image — no path edits). Docking is deterministic (`--random_seed 2023`)

> The evaluation set is **[`data/proto_test.csv`](data/proto_test.csv) (136 complexes)**. The reference structures come from the bundle. `proto_train` (712) is unchanged.

### Retraining from scratch and fine-tuning
Hyper-parameters are in [§5](#5-training-information--parameters-from-the-paper); the cluster drivers are [`condor/p2_train_scratch.sub`](condor/p2_train_scratch.sub) (P2) and
[`condor/p3_finetune.sub`](condor/p3_finetune.sub) (P3). Training is the long path (on one GPU **~54 hours** for the 712 complexes in proto_train).

## 5. Training information & parameters (from the paper)

[`train.py`](scripts/train.py) implements the KarmaDock paper's *Training protocol*. Effective batch = 64 in all runs
(`batch_size 4 × accum_steps 16`); train/val split is deterministic (`val_frac 0.1`, `seed 42`).

**P2 — from scratch (2 stages, paper protocol):**

| stage | objective | `pos_r` | optimizer | lr | weight_decay | patience |
|---|---|---|---|---|---|---|
| 1 | scoring / MDN only (EGNN docking skipped) | 0 | Adam | 1e-3 | 1e-5 | 70 |
| 2 | docking + scoring (init from Stage-1 best) | 1 | Adam | 1e-4 | 1e-4 | 20 |

**P3 — fine-tune (bonus, single stage, init = released weights):**
`pos_r 1`, Adam, `lr 1e-4`, `weight_decay 0`, `patience 30`, eff. batch 64, `val_frac 0.1`, `seed 42`.

**pos_r** is the scalar weight on the RMSD (coordinate/docking) loss in KarmaDock's `training
  objective loss = pos_r * rmsd_loss + mdn_loss` — it acts as a positional-refinement switch
  that is set to 0 in Stage 1 (train only the MDN interaction-distance loss) and 1 in
  Stage 2 (turn on the RMSD term to refine predicted ligand coordinates toward the crystal
  pose).

Per-epoch training curves are in [`docs/p2_stage1_train_log.csv`](docs/p2_stage1_train_log.csv),
[`docs/p2_stage2_train_log.csv`](docs/p2_stage2_train_log.csv) and [`docs/p3_finetune_train_log.csv`](docs/p3_finetune_train_log.csv).

## 6. Repository layout

| path | description |
|---|---|
| [`README.md`](README.md) | this file |
| [`Dockerfile`](Dockerfile) | image  (`ahlamloum/karmadock-seminar:v6`) |
| [`scripts/`](scripts) | `train.py` (main artifact), converters, `run_infer.sh`, `evaluate.sh`, `run_train.sh` |
| [`condor/`](condor) | portable HTCondor submit files (3 docking + 1 eval + 2 training) |
| [`evaluation/evaluation.py`](evaluation/evaluation.py) | the seminar's official evaluator (unmodified) |
| [`model/`](model) | P2 + P3 trained checkpoints (~15 MB each) |
| [`notebooks/results_and_comparison.ipynb`](notebooks/results_and_comparison.ipynb) | tables, charts |
| [`results/`](results) | `proto_test/` = P2 poses that `evaluation.py --dataset proto_test` scores; `<pipeline>/proto_test{,_ff,_align}/` = all 3 pipelines × 3 variants; `*_evaluation.csv` = official RMSD per pipeline × variant |
| [`docs/`](docs) | training logs + figures |
| [`data/proto_test.csv`](data/proto_test.csv) | proto_test mapping (136 complexes) |
| [`data/prototype_model_data.zip`](data/prototype_model_data.zip) | reference structures: proto_test + proto_train, refined SDF/PDB |
| [`scripts/README.md`](scripts/README.md) | provenance: our code vs. upstream KarmaDock |

**Docker image:** `ahlamloum/karmadock-seminar:v6` (Docker Hub).


## 7. Issues & fixes

- **Node Problem**: in the idun cluster the gpu nodes are sometime during the run just return error so we had to exclude this specific node as a requirement in our sub files.
Fix: exclude it —
  `requirements = … && (Machine =!= "idun.hpc.uni-saarland.de")`.
 **[solved]**.
 - resources limitations: `request_gpus=2/4` jobs sat idle for days. The prototype runs single-GPU. we couldn't run on 2 gpus that's why we changed it to 1 gpu, we are not sure if this gonna work on the full dataset as it will take too much time.
Fix: change the request to 1 gpu
 **[solved]**.
