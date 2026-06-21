# Results

Predicted poses and official evaluations on the corrected **136-complex `proto_test`**.

## Layout

```
results/
├── proto_test/                         primary set = from-scratch P2, UNCORRECTED poses (136)
│                                        (what `python evaluation/evaluation.py --dataset proto_test` scores)
├── p1_baseline/  proto_test/  proto_test_ff/  proto_test_align/    136 each
├── p2_scratch/   proto_test/  proto_test_ff/  proto_test_align/    136 each
├── p3_finetune/  proto_test/  proto_test_ff/  proto_test_align/    136 each
├── proto_test_evaluation.csv           official RMSD for the primary set (= p2_scratch uncorrected)
├── <pipeline>_<variant>_evaluation.csv per-complex RMSD for each of the 9 pipeline x variant combos
└── README.md
```

Pose files are `<complex_id>_pred.sdf`, conformers ranked best-MDN-first. The three variants are the
KarmaDock pose post-processing options: **uncorrected** (raw network output), **FF** (force-field
relaxation), **align** (superimpose onto the reference frame).

## Eval CSV columns

`complex_id, dataset, pose_rank, rmsd, rmsd_lt2, rmsd_lt1, ligand_file, protein_file`
(RDKit `GetBestRMS`, symmetry-corrected, top-1). `success@2Å` = fraction of `pose_rank == 1` rows
with `rmsd_lt2 == True`.

## success@2 Å (top-1)

| pipeline | uncorrected | FF | align |
|---|---|---|---|
| **P2 — from scratch** (primary) | 10.3 % | 11.0 % | 94.1 % |
| P1 — baseline (released) | 80.9 % | 78.7 % | 95.6 % |
| P3 — fine-tune (bonus) | 80.1 % | 75.0 % | 94.9 % |

> The **align** variant superimposes the predicted ligand onto the reference frame, so on this
> re-docking set it scores ~94–96 % for every model regardless of docking quality (the from-scratch
> P2, ~10 % uncorrected, also reaches 94 % aligned). It reflects the reference, not docking skill.

## Reproduce

```bash
unzip -o ../data/prototype_model_data.zip -d ../data/prototype_model_data
ln -sf prototype_model_data/proto_test ../data/proto_test
python ../evaluation/evaluation.py --dataset proto_test   # scores results/proto_test/ (= P2)
```
To regenerate the poses on the cluster, see the top-level `README.md` §4.
