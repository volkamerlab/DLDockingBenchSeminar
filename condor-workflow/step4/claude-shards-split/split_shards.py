#!/usr/bin/env python3
"""
split_shards.py -- partition InterFormer docking input into N self-contained shards.

WHY
    A single HTCondor job can't hold the whole dataset. The transferred input
    alone was ~92 GiB and filled the execute node's ~95 GiB scratch, so docking
    silently stopped after ~2,018 of 25,939 ligands (no ENOSPC traceback --
    PyVina just wrote fewer poses). Splitting the FOUR input folders into N
    shards keeps each job at ~(92/N) GiB + its own output, well inside the disk
    cap, and runs the shards in parallel (cutting wall-clock too).

KEY FACT
    The docking step (`reconstruct_ligands.py --find_all find`) discovers ligands
    by SCANNING the folders, not from a CSV. So we only need to shard the folders.
    The experimental-label CSV is merged once at the very end (see merge_shards.py),
    which is also why we don't touch it here -- no dependency on its column schema.

HOW
    Ligand identity ("pdb_id", e.g. 1xmy_ROL_A_101) is taken from the canonical
    gaussian_predict/<pdb_id>_G.db files. Every file in complex/ ligand/ uff/
    whose name begins with that pdb_id (at a non-alphanumeric boundary) is grouped
    into the SAME shard, so each shard is a consistent, dockable subset.

USAGE
    # sanity-check the split first (fast, writes only the manifest, no tarballs):
    python split_shards.py --data-dir dock_results/energy_train --num-shards 10 --dry-run

    # then build the shards:
    python split_shards.py --data-dir dock_results/energy_train \
                           --out-dir shards --num-shards 10

OUTPUTS
    shards/shard_<i>/complex.tar.gz gaussian_predict.tar.gz ligand.tar.gz uff.tar.gz
    shards/manifest.csv    (shard, n_ligands, and per-folder file counts)
    shards/unmatched.txt   (files that matched no pdb_id, if any -- investigate these)

NOTE
    Run this on a node that HAS the extracted folders and enough room (your submit
    node / project dir). If you only have the *.tar.gz here, extract them once first:
        for t in complex gaussian_predict ligand uff; do
            tar -xzf "$t.tar.gz" -C dock_results/energy_train; done
"""
import argparse
import bisect
import os
import sys
import tarfile


def enumerate_ligand_ids(gauss_dir, gaussian_suffix="_G.db"):
    """Canonical pdb_id list from gaussian_predict/<pdb_id>_G.db (fallback: *.db)."""
    ids = set()
    if not os.path.isdir(gauss_dir):
        sys.exit(f"[split_shards] gaussian folder not found: {gauss_dir}\n"
                 f"               (did you extract the tarballs into --data-dir?)")
    with os.scandir(gauss_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            n = e.name
            if n.endswith(gaussian_suffix):
                ids.add(n[: -len(gaussian_suffix)])
            elif n.endswith(".db"):
                ids.add(n[:-3])
    if not ids:
        sys.exit(f"[split_shards] no *{gaussian_suffix} files in {gauss_dir}")
    return ids


def make_id_lookup(ids):
    """Return (sorted_ids, fn) where fn(filename)->owning pdb_id or None.

    Uses bisect to find the LONGEST id that is a prefix of the filename, ending
    at a non-alphanumeric boundary. This correctly disambiguates ids that are
    prefixes of one another (e.g. a file '1a08_x' maps to id '1a08', never '1a0')."""
    ids_sorted = sorted(ids)

    def id_for(fname):
        j = bisect.bisect_right(ids_sorted, fname) - 1
        if j < 0:
            return None
        cand = ids_sorted[j]
        if fname.startswith(cand):
            nxt = fname[len(cand):len(cand) + 1]
            if nxt == "" or not nxt.isalnum():
                return cand
        return None

    return ids_sorted, id_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="dock_results/energy_train",
                    help="folder containing extracted complex/ gaussian_predict/ ligand/ uff/")
    ap.add_argument("--out-dir", default="shards",
                    help="where to write shard_<i>/ tarballs + manifest")
    ap.add_argument("--num-shards", type=int, required=True,
                    help="number of shards == number of HTCondor jobs (queue N)")
    ap.add_argument("--folders", default="complex,gaussian_predict,ligand,uff",
                    help="comma-separated input folders to shard")
    ap.add_argument("--gaussian-suffix", default="_G.db",
                    help="suffix that marks a canonical ligand in gaussian_predict/")
    ap.add_argument("--dry-run", action="store_true",
                    help="only compute + write the manifest; do not create tarballs")
    args = ap.parse_args()

    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    N = args.num_shards
    if N < 1:
        sys.exit("[split_shards] --num-shards must be >= 1")

    gauss_dir = os.path.join(args.data_dir, "gaussian_predict")
    ids = enumerate_ligand_ids(gauss_dir, args.gaussian_suffix)
    ids_sorted, id_for = make_id_lookup(ids)
    print(f"[split_shards] {len(ids)} ligand ids; splitting into {N} shards "
          f"(round-robin, ~{len(ids)//N} ligands/shard)")

    # round-robin assignment keeps shard sizes balanced and mixes any ordering
    shard_of_id = {pid: i % N for i, pid in enumerate(sorted(ids))}

    # walk each folder ONCE; bucket every file by (shard, folder)
    buckets = {}                       # (shard, folder) -> [absolute paths]
    counts = [{f: 0 for f in folders} for _ in range(N)]
    unmatched = []
    for folder in folders:
        fdir = os.path.join(args.data_dir, folder)
        if not os.path.isdir(fdir):
            print(f"[split_shards] WARNING: missing folder, skipping: {fdir}",
                  file=sys.stderr)
            continue
        n_seen = 0
        with os.scandir(fdir) as it:
            for e in it:
                if not e.is_file():
                    continue
                n_seen += 1
                pid = id_for(e.name)
                if pid is None:
                    unmatched.append(f"{folder}/{e.name}")
                    continue
                s = shard_of_id[pid]
                buckets.setdefault((s, folder), []).append(e.path)
                counts[s][folder] += 1
        print(f"[split_shards]   {folder}: {n_seen} files scanned")

    os.makedirs(args.out_dir, exist_ok=True)

    # manifest + unmatched report (always written)
    manifest = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest, "w") as fh:
        fh.write("shard,n_ligands," + ",".join(folders) + "\n")
        n_ids_per_shard = [0] * N
        for pid, s in shard_of_id.items():
            n_ids_per_shard[s] += 1
        for s in range(N):
            row = [str(s), str(n_ids_per_shard[s])] + [str(counts[s][f]) for f in folders]
            fh.write(",".join(row) + "\n")
    print(f"[split_shards] wrote {manifest}")

    if unmatched:
        upath = os.path.join(args.out_dir, "unmatched.txt")
        with open(upath, "w") as fh:
            fh.write("\n".join(unmatched) + "\n")
        print(f"[split_shards] WARNING: {len(unmatched)} files matched no pdb_id "
              f"-> {upath} (these will NOT be docked; check naming)")

    if args.dry_run:
        print("[split_shards] --dry-run: manifest only, no tarballs written.")
        return

    # build the per-shard tarballs
    for s in range(N):
        sdir = os.path.join(args.out_dir, f"shard_{s}")
        os.makedirs(sdir, exist_ok=True)
        for folder in folders:
            paths = buckets.get((s, folder), [])
            out_tar = os.path.join(sdir, f"{folder}.tar.gz")
            with tarfile.open(out_tar, "w:gz") as tf:
                for p in paths:
                    # arcname keeps the folder/ prefix so extraction into the job's
                    # DOCK_FOLDER recreates dock_results/energy_train/<folder>/<file>
                    tf.add(p, arcname=f"{folder}/{os.path.basename(p)}")
        print(f"[split_shards] shard_{s}: "
              + ", ".join(f"{folder}={counts[s][folder]}" for folder in folders))

    print(f"[split_shards] done. Set 'queue {N}' in the .sub to match.")


if __name__ == "__main__":
    main()
