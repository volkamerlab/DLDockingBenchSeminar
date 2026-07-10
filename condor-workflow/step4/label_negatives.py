"""
Ben's Notes: This is a hardened version of changing the pIC50 value to a negative value for other downstream processes.
Note that when calling this script, and no additional arguments are provided, --cutoff 's value is 2.0 and --guarantee-positive defaults to False.

Claude:

label_negatives.py
-------------------
Applies InterFormer's contrastive pose labeling to a merged round0 CSV.

The affinity+pose model (`-filter_type full`) expects the training CSV to already
contain SIGNED pIC50 values:
    - a pose that reproduces the native binding mode (rmsd <= cutoff) keeps its
      positive experimental pIC50   ->  positive sample
    - a decoy pose (rmsd > cutoff)  gets its pIC50 negated  ->  negative sample

BindingData._filter_df in 'full' mode uses threshold=-inf, so it PRESERVES these
negatives; shard_dataset.make_shared then routes y<=0 rows into the *.neg.bin
shards. Nothing downstream creates the sign, so it must be written here.

Usage:
    python label_negatives.py input_round0.csv [output.csv] \

    --cutoff              RMSD (A) at/below which a pose is treated as native.
    --guarantee-positive  Keep each Target's lowest-rmsd pose positive even if it
                          exceeds the cutoff, so no ligand ends up all-negative
                          (which would give that target an empty 'pos' shard).
"""
import argparse
import sys
import pandas as pd


def label(df, cutoff=2.0, guarantee_positive=False,
          label_key="pIC50", rmsd_key="rmsd", target_key="Target"):
    for col in (label_key, rmsd_key, target_key):
        if col not in df.columns:
            sys.exit(f"[label_negatives] Missing required column: '{col}'")

    df[label_key] = df[label_key].astype(float)
    df[rmsd_key] = df[rmsd_key].astype(float)

    # Work from the magnitude so re-running the script is idempotent.
    mag = df[label_key].abs()
    is_native = df[rmsd_key] <= cutoff

    if guarantee_positive:
        # Mark each target's best (lowest-rmsd) pose as native regardless of cutoff.
        best_idx = df.groupby(target_key)[rmsd_key].idxmin()
        is_native.loc[best_idx] = True

    df[label_key] = mag.where(is_native, -mag)

    n_pos = int((df[label_key] > 0).sum())
    n_neg = int((df[label_key] < 0).sum())
    n_targets_all_neg = int(
        (df.groupby(target_key)[label_key].max() <= 0).sum()
    )
    print(f"[label_negatives] cutoff={cutoff} A, guarantee_positive={guarantee_positive}")
    print(f"[label_negatives] positive rows: {n_pos}  negative rows: {n_neg}  "
          f"({100 * n_neg / max(len(df), 1):.1f}% negative)")
    if n_targets_all_neg:
        print(f"[label_negatives] WARNING: {n_targets_all_neg} target(s) have NO "
              f"positive pose. Consider --guarantee-positive or add the crystal "
              f"(rmsd~0) reference pose per target.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv", nargs="?", default=None)
    ap.add_argument("--cutoff", type=float, default=2.0)
    ap.add_argument("--guarantee-positive", action="store_true")
    args = ap.parse_args()

    out = args.output_csv or args.input_csv.replace(".csv", ".labeled.csv")
    df = pd.read_csv(args.input_csv, low_memory=False)
    df = label(df, cutoff=args.cutoff, guarantee_positive=args.guarantee_positive)
    df.to_csv(out, index=False)
    print(f"[label_negatives] wrote -> {out}")


if __name__ == "__main__":
    main()