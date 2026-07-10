#!/usr/bin/env python3
"""
merge_shards.py -- reassemble sharded docking output and finish the pipeline.

Run this ONCE on the submit node after all shard jobs finish. It:
  1. extracts every out_shard_*.tar.gz, consolidating all SDF poses into a single
     results/ligand_reconstructing/ and keeping each shard's stat CSV separately;
  2. concatenates the per-shard stat CSVs -> results/ligand_reconstructing/stat_concated.csv
     (this is the exact path the original merge step expects);
  3. (optional, --finish) runs merge_summary_input.py + label_negatives.py once on
     the combined result -- the steps that were per-shard-broken before.

Step 3 is what previously crashed: the job looked for data/proto_train_val_final.csv
(which was never transferred). Here it runs post-shard, against the full CSV you point
it at, so the sign/label logic sees the whole dataset at once.

USAGE
    python merge_shards.py --shards-glob 'out_shard_*.tar.gz' --out-dir results
    # then, to also run merge + label (needs the full experimental CSV present):
    python merge_shards.py --shards-glob 'out_shard_*.tar.gz' --out-dir results \
        --finish --orig-csv proto_train_val_final.csv --docking-dir docking \
        --label-script label_negatives.py
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile


def safe_move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):          # same ligand name across shards shouldn't happen; keep first
        return
    try:
        shutil.move(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def extract_and_consolidate(tarballs, out_dir):
    recon_dir = os.path.join(out_dir, "ligand_reconstructing")
    stats_dir = os.path.join(out_dir, "stats")
    tmp_root = os.path.join(out_dir, "_tmp")
    for d in (recon_dir, stats_dir, tmp_root):
        os.makedirs(d, exist_ok=True)

    stat_files = []
    for tb in sorted(tarballs):
        # shard index from out_shard_<i>.tar.gz
        base = os.path.basename(tb)
        idx = base.replace("out_shard_", "").replace(".tar.gz", "")
        tmp = os.path.join(tmp_root, f"shard_{idx}")
        os.makedirs(tmp, exist_ok=True)
        print(f"[merge_shards] extracting {tb}")
        with tarfile.open(tb, "r:gz") as tf:
            tf.extractall(tmp)                      # -> tmp/ligand_reconstructing/...

        src_recon = os.path.join(tmp, "ligand_reconstructing")
        if not os.path.isdir(src_recon):
            print(f"[merge_shards] WARNING: no ligand_reconstructing/ in {tb}", file=sys.stderr)
            continue
        for name in os.listdir(src_recon):
            src = os.path.join(src_recon, name)
            if name == "stat_concated.csv":
                dst = os.path.join(stats_dir, f"stat_shard_{idx}.csv")
                safe_move(src, dst)
                stat_files.append(dst)
            else:
                # SDF poses (and any per-ligand artifacts) -> one consolidated folder
                safe_move(src, os.path.join(recon_dir, name))
    shutil.rmtree(tmp_root, ignore_errors=True)
    return recon_dir, stat_files


def concat_stats(stat_files, dest_csv):
    """Concatenate per-shard stat CSVs, keeping a single header row."""
    if not stat_files:
        sys.exit("[merge_shards] no stat_concated.csv found in any shard -- did the jobs run?")
    n_rows = 0
    with open(dest_csv, "w", newline="") as out:
        header_written = False
        for i, sf in enumerate(sorted(stat_files)):
            with open(sf, "r", newline="") as fh:
                lines = fh.readlines()
            if not lines:
                continue
            if not header_written:
                out.write(lines[0])
                header_written = True
            body = lines[1:]
            n_rows += len(body)
            out.writelines(body)
    print(f"[merge_shards] wrote {dest_csv}  ({n_rows} data rows from {len(stat_files)} shards)")
    return dest_csv


def run(cmd):
    print("[merge_shards] $ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def finish_pipeline(stat_all, orig_csv, docking_dir, label_script):
    """Run merge_summary_input.py then label_negatives.py, robust to unknown output names."""
    merge_py = os.path.join(docking_dir, "merge_summary_input.py")
    for p in (merge_py, orig_csv, label_script):
        if not os.path.exists(p):
            print(f"[merge_shards] --finish skipped: missing {p}", file=sys.stderr)
            _print_manual(stat_all, orig_csv, merge_py, label_script)
            return

    before = set(glob.glob("*.csv"))
    rc = run([sys.executable, merge_py, stat_all, orig_csv])
    if rc != 0:
        print("[merge_shards] merge_summary_input.py returned non-zero; check its args.",
              file=sys.stderr)
    # detect what merge produced (its output filename isn't documented here)
    new_csvs = sorted(set(glob.glob("*.csv")) - before,
                      key=lambda p: os.path.getmtime(p), reverse=True)
    round0 = next((c for c in new_csvs if "round0" in c), (new_csvs[0] if new_csvs else None))
    if not round0:
        print("[merge_shards] could not detect merge output CSV; run label step manually.",
              file=sys.stderr)
        _print_manual(stat_all, orig_csv, merge_py, label_script)
        return
    print(f"[merge_shards] merge output detected: {round0}")

    labeled = round0.replace(".csv", "_w_neg_labels.csv")
    run([sys.executable, label_script, round0, labeled, "--guarantee-positive"])
    print(f"[merge_shards] done -> {labeled}")


def _print_manual(stat_all, orig_csv, merge_py, label_script):
    print("\n[merge_shards] run these two steps manually:")
    print(f"    python {merge_py} {stat_all} {orig_csv}")
    print(f"    python {label_script} <merge_output>.csv "
          f"<merge_output>_w_neg_labels.csv --guarantee-positive\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards-glob", default="out_shard_*.tar.gz",
                    help="glob for the shard output tarballs")
    ap.add_argument("--out-dir", default="results",
                    help="where to reassemble ligand_reconstructing/ + stats/")
    ap.add_argument("--finish", action="store_true",
                    help="also run merge_summary_input.py + label_negatives.py")
    ap.add_argument("--orig-csv", default="proto_train_val_final.csv",
                    help="full experimental-label CSV (used by --finish)")
    ap.add_argument("--docking-dir", default="docking",
                    help="dir containing merge_summary_input.py (used by --finish)")
    ap.add_argument("--label-script", default="label_negatives.py",
                    help="path to label_negatives.py (used by --finish)")
    args = ap.parse_args()

    tarballs = glob.glob(args.shards_glob)
    if not tarballs:
        sys.exit(f"[merge_shards] no shard tarballs matched {args.shards_glob!r}")
    print(f"[merge_shards] {len(tarballs)} shard tarballs")

    os.makedirs(args.out_dir, exist_ok=True)
    recon_dir, stat_files = extract_and_consolidate(tarballs, args.out_dir)
    n_sdf = sum(1 for n in os.listdir(recon_dir) if n.endswith(".sdf"))
    print(f"[merge_shards] consolidated {n_sdf} SDF files into {recon_dir}")

    stat_all = os.path.join(recon_dir, "stat_concated.csv")
    concat_stats(stat_files, stat_all)

    if args.finish:
        finish_pipeline(stat_all, args.orig_csv, args.docking_dir, args.label_script)
    else:
        print("[merge_shards] stats concatenated. Re-run with --finish to merge + label,")
        print("               or run docking/merge_summary_input.py yourself on:")
        print(f"                 {stat_all}")


if __name__ == "__main__":
    main()
