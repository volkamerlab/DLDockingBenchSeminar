#!/usr/bin/env python3
"""
build_full_test_csv.py

Build a `full_test.csv` mapping file for the docking-evaluation pipeline.

It scans a results directory full of predicted-pose SDF files
(e.g. results/full_test/1a0t_GLC-FRU_A_1-2_pred.sdf), derives the complex
prefix from each filename, and writes one row per complex with:

    ligand_file_name   = "{prefix}_ligand_refined.sdf"
    protein_file_name  = "{prefix}_protein_refined.pdb"

It then looks the prefix up in a reference CSV (proto_test_final.csv) via the
"Target" column and copies the following metadata columns:

    Year, Log Binding Affinity, Binding Affinity Measurement, PDBID

Prefixes with no match in the reference CSV are kept by default (metadata left
blank) so that every predicted pose still gets evaluated; use --drop-unmatched
to exclude them instead.

Output column order matches what evaluation.py expects for proto_test/full_test:
    ligand_file_name,protein_file_name,Year,Log Binding Affinity,Binding Affinity Measurement,PDBID
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Filename convention
PRED_SUFFIX = "_pred.sdf"
LIGAND_SUFFIX = "_ligand_refined.sdf"
PROTEIN_SUFFIX = "_protein_refined.pdb"

# Metadata columns pulled from the reference CSV, keyed on its "Target" column
TARGET_COL = "Target"
META_COLS = ["Year", "Log Binding Affinity", "Binding Affinity Measurement", "PDBID"]

# Final output column order (must match evaluation.py's expected format)
OUTPUT_COLS = ["ligand_file_name", "protein_file_name"] + META_COLS


def get_prefix(sdf_name: str) -> str:
    """'1a0t_GLC-FRU_A_1-2_pred.sdf' -> '1a0t_GLC-FRU_A_1-2'."""
    if sdf_name.endswith(PRED_SUFFIX):
        return sdf_name[: -len(PRED_SUFFIX)]
    return Path(sdf_name).stem  # fallback: just strip the extension


def collect_prefixes(results_dir: Path) -> list[str]:
    """Return sorted, de-duplicated prefixes from every *_pred.sdf in results_dir."""
    if not results_dir.is_dir():
        sys.exit(f"ERROR: results directory not found: {results_dir}")

    sdf_files = sorted(results_dir.glob(f"*{PRED_SUFFIX}"))
    if not sdf_files:
        sys.exit(f"ERROR: no '*{PRED_SUFFIX}' files found in {results_dir}")

    prefixes = [get_prefix(p.name) for p in sdf_files]
    # de-dup while preserving order
    seen, unique = set(), []
    for pfx in prefixes:
        if pfx not in seen:
            seen.add(pfx)
            unique.append(pfx)
    return unique


def load_reference(ref_csv: Path) -> pd.DataFrame:
    """Load reference CSV, keep Target + metadata columns, drop duplicate Targets."""
    if not ref_csv.is_file():
        sys.exit(f"ERROR: reference CSV not found: {ref_csv}")

    df = pd.read_csv(ref_csv, dtype=str)  # read as str to preserve values exactly

    missing = [c for c in [TARGET_COL] + META_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: reference CSV is missing columns: {missing}")

    df = df[[TARGET_COL] + META_COLS].copy()

    dupes = df[TARGET_COL].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate '{TARGET_COL}' value(s) in reference CSV; keeping first.")
        df = df.drop_duplicates(subset=TARGET_COL, keep="first")

    return df


def build(results_dir: Path, ref_csv: Path, drop_unmatched: bool) -> pd.DataFrame:
    prefixes = collect_prefixes(results_dir)
    print(f"Found {len(prefixes)} unique complex prefix(es) in {results_dir}")

    base = pd.DataFrame({"prefix": prefixes})
    base["ligand_file_name"] = base["prefix"] + LIGAND_SUFFIX
    base["protein_file_name"] = base["prefix"] + PROTEIN_SUFFIX

    ref = load_reference(ref_csv)

    merged = base.merge(ref, how="left", left_on="prefix", right_on=TARGET_COL)

    n_matched = merged[TARGET_COL].notna().sum()
    n_unmatched = len(merged) - n_matched
    print(f"Matched {n_matched} against '{TARGET_COL}' in {ref_csv.name}; {n_unmatched} unmatched.")

    if n_unmatched:
        examples = merged.loc[merged[TARGET_COL].isna(), "prefix"].head(5).tolist()
        print(f"  Example unmatched prefixes: {examples}")
        if drop_unmatched:
            merged = merged[merged[TARGET_COL].notna()].copy()
            print(f"  Dropped {n_unmatched} unmatched row(s) (--drop-unmatched).")
        else:
            print("  Keeping unmatched rows with blank metadata (use --drop-unmatched to exclude).")

    return merged[OUTPUT_COLS].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build full_test.csv from predicted-pose SDF filenames + reference metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=Path("results/full_test"),
                   help="Directory containing *_pred.sdf files.")
    p.add_argument("--ref-csv", type=Path, default=Path("proto_test_final.csv"),
                   help="Reference CSV with a 'Target' column and metadata.")
    p.add_argument("--output-csv", type=Path, default=Path("full_test.csv"),
                   help="Output CSV path.")
    p.add_argument("--drop-unmatched", action="store_true",
                   help="Exclude prefixes with no match in the reference CSV "
                        "(default: keep them with blank metadata).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build(args.results_dir, args.ref_csv, args.drop_unmatched)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()