# Sharded InterFormer docking (HTCondor)

## Why this exists

The single-job run docked only **2,018 of 25,939 ligands**. The cause was **disk, not compute**:

| resource | used / granted | at end of run |
| --- | --- | --- |
| disk | 92 GiB / ~95 GiB (**96.7%**) | pinned — this stopped it |
| memory | 26 GB / 128 GB (20%) | idle headroom |
| GPU memory | 521 MB / 40 GB (1.3%) | idle headroom |
| CPU | 0 of 32 at eviction | docking already stopped |

The ~92 GiB of transferred input filled the execute node's scratch, so PyVina
silently wrote fewer poses once the disk was full (that's why there's no `ENOSPC`
in the log). Adding CPU/GPU could not have helped. Sharding fixes the real limit:
each job now handles ~`1/N` of the data, so its footprint stays well inside the cap,
and the shards run in parallel (also cutting the ~5-day single-job wall-clock).

The separate *crash* at the end was the merge step looking for
`data/proto_train_val_final.csv` (never transferred), which then made the
`transfer_output_files` for `proto_train_val_final.round0.csv` fail. That step is
moved out of the per-shard job and run once at the end (`merge_shards.py`).

## Files

- `split_shards.py` — splits the four input folders into `N` self-contained shard tarballs.
- `interformer_affinity_pose_docking.sub` — HTCondor submit: `queue N`, `$(Process)`, 3 bugs fixed.
- `interformer_affinity_pose_docking.sh` — per-shard job: robust unpack + `find` + `stat`, packages its own output.
- `merge_shards.py` — reassembles shard outputs, then runs merge + label once.

## Run order

**1. Build the shards** (on a node that has the extracted folders + room; e.g. submit node).
If you only have the `*.tar.gz`, extract them once into `dock_results/energy_train/` first.

```bash
# sanity-check the split (fast, manifest only):
python split_shards.py --data-dir dock_results/energy_train --num-shards 10 --dry-run
# inspect shards/manifest.csv and shards/unmatched.txt, then build for real:
python split_shards.py --data-dir dock_results/energy_train --out-dir shards --num-shards 10
```

**2. Submit `N` docking jobs.** Set `queue N` in the `.sub` to the same number, then:

```bash
mkdir -p logs
condor_submit interformer_affinity_pose_docking.sub
```

Each job returns `out_shard_<i>.tar.gz` (SDF poses + that shard's `stat_concated.csv`).

**3. Reassemble + finish** once all shards complete:

```bash
python merge_shards.py --shards-glob 'out_shard_*.tar.gz' --out-dir results \
    --finish --orig-csv proto_train_val_final.csv \
    --docking-dir docking --label-script label_negatives.py
```

Output: `results/ligand_reconstructing/` (all poses + combined `stat_concated.csv`)
and `*.round0_w_neg_labels.csv`. Drop `--finish` to stop after concatenating stats
and run the merge/label steps yourself.

## Choosing N and `request_disk`

- Input is ~92 GiB total, so each shard transfers ~`92/N` GiB. **N = 10** → ~9 GiB
  input + a few GiB of poses per job. `request_disk = 50G` in the `.sub` is safe for that.
- Fewer shards → raise `request_disk`. Keep it **≤ ~95 GiB** (what the node actually
  grants). The old `.sub` asked for `200G` but was only given ~95 GiB — worth raising
  with your admins if you need bigger jobs, but sharding sidesteps it.
- More shards → smaller/faster jobs, more scheduling overhead. 8–12 is a good range.

## Bugs fixed vs. the original `.sub`

1. `transfer_input_files = transfer_input_files = ...` (doubled token) → single assignment.
2. Two `requirements =` lines (the second silently overrode the A100 constraint) → one combined expression.
3. Tarballs landed in `dock_results/energy_train/` but the script looked in `./`, so the
   unpack loop never fired → the `.sh` now finds the tarballs wherever HTCondor drops them.
4. The job no longer expects `proto_train_val_final.round0.csv` as output (nothing in the
   docking step creates it); merge/label runs once, post-shard.
