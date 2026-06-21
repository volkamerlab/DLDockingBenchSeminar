# PROVENANCE — what we wrote vs. the KarmaDock authors'

**The key point:** public KarmaDock (`schrojunzhang/KarmaDock`) ships **inference code +
pretrained weights only — there is NO training script.** Our seminar contribution is the
**training loop, the data adapters, the run wrappers, and the whole HTCondor orchestration**
built *around* KarmaDock. We **call KarmaDock's modules as-is** (model, preprocessing, docking)
and did **not modify any file inside the upstream `KarmaDock/`** — it is cloned fresh in the
`Dockerfile`; all of our code lives in `scripts/`.

## 1. Files WE created (our original work)

### `scripts/`
| file | what it does | KarmaDock pieces it calls |
|---|---|---|
| `train.py` | **our main artifact** — a complete checkpointed training/fine-tuning loop (KarmaDock has none): loop, optimizer, early-stopping, gradient accumulation, val split, W&B, `--resume`. | imports the upstream `KarmaDock` model, `PDBBindGraphDataset`, `PassNoneDataLoader`, `set_random_seed`/`Early_stopper`; uses the model's own `forward()` losses |
| `convert_seminar_to_karmadock.py` | seminar data layout → KarmaDock layout | — |
| `convert_karmadock_to_seminar.py` | KarmaDock docked poses → seminar `<id>_pred.sdf` (best-first) | reads pose SDFs (RDKit) |
| `run_infer.sh` | **portable** docking: preprocess → dock → export the 3 pose variants (uncorrected/FF/align) | calls upstream `pre_processing.py`, `generate_graph.py`, `ligand_docking.py` + our converters |
| `evaluate.sh` | the separate scoring step: runs `evaluation.py` over every pipeline × variant | calls the seminar's `evaluation/evaluation.py` |
| `run_train.sh` | **portable** training: `scratch` = paper 2-stage (P2), `finetune` = from released weights (P3); preprocesses `proto_train` first | calls our `train.py` |

### `condor/` (our work)
| file | job |
|---|---|
| `p1_baseline.sub` | P1 inference (released weights) + eval |
| `p2_scratch_infer.sub` | P2 inference (our from-scratch checkpoint) + eval |
| `p3_finetune_infer.sub` | P3 inference (our fine-tuned checkpoint) + eval |
| `p2_train_scratch.sub` | P2 training (2-stage from scratch) |
| `p3_finetune.sub` | P3 fine-tune training |

### Other ours
- `Dockerfile` — builds `ahlamloum/karmadock-seminar:v6` (clones KarmaDock, installs the authors' packed conda env, adds our `scripts/`).
- `model/` checkpoints (P2, P3), the results notebook, all docs, and the training logs in `docs/`.

## Summary
- **Authors':** the KarmaDock model + preprocessing/docking utilities + released weights (used as-is).
- **Ours:** `train.py` (there was no trainer), the data adapters, the run wrappers, every condor
  `.sub`, the Docker build, and all analysis/docs. **No KarmaDock source file was edited.**
