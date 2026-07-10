import sys
import pickle
import lmdb
from collections import defaultdict


def check_lmdb(lmdb_path, max_print=20):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    n_total = 0
    n_empty_coords = 0
    n_atom_mismatch = 0
    bad_keys = []
    smi_groups = defaultdict(list)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            n_total += 1
            entry = pickle.loads(value)

            coords = entry.get("coordinates", [])
            atoms = entry.get("atoms", [])
            smi = entry.get("smi", None)

            n_confs = len(coords) if coords is not None else 0
            smi_groups[smi].append((key.decode(), n_confs))

            if n_confs == 0:
                n_empty_coords += 1
                bad_keys.append((key.decode(), "zero conformers", entry.get("pocket", "?")))
                continue

            n_atoms = len(atoms)
            for ci, conf in enumerate(coords):
                if len(conf) != n_atoms:
                    n_atom_mismatch += 1
                    bad_keys.append((
                        key.decode(),
                        "conformer %d has %d coords vs %d atoms" % (ci, len(conf), n_atoms),
                        entry.get("pocket", "?"),
                    ))
                    break

    env.close()

    inconsistent_groups = 0
    for smi, items in smi_groups.items():
        counts = set(c for _, c in items)
        if len(items) > 1 and len(counts) > 1:
            inconsistent_groups += 1

    print("")
    print("=== %s ===" % lmdb_path)
    print("Total entries:                 %d" % n_total)
    print("Entries with 0 conformers:     %d" % n_empty_coords)
    print("Entries with atom/coord mismatch: %d" % n_atom_mismatch)
    print("Duplicate-SMILES groups with inconsistent conformer counts: %d / %d groups" % (inconsistent_groups, len(smi_groups)))

    if bad_keys:
        print("")
        print("First %d problem entries (key, pocket, reason):" % min(max_print, len(bad_keys)))
        for k, reason, pocket in bad_keys[:max_print]:
            print("  key=%s  pocket=%s  -> %s" % (k, pocket, reason))
        if len(bad_keys) > max_print:
            print("  ... and %d more" % (len(bad_keys) - max_print))
    else:
        print("")
        print("No malformed entries found.")

    return bad_keys


def write_clean_lmdb(src_path, dst_path, bad_keys):
    bad_key_set = set(k for k, _, _ in bad_keys)
    env_src = lmdb.open(src_path, subdir=False, readonly=True, lock=False)
    env_dst = lmdb.open(dst_path, subdir=False, map_size=int(1e10))

    count = 0
    with env_src.begin() as txn_src, env_dst.begin(write=True) as txn_dst:
        for key, value in txn_src.cursor():
            if key.decode() in bad_key_set:
                continue
            txn_dst.put(("%d" % count).encode("ascii"), value)
            count += 1

    env_src.close()
    env_dst.close()
    print("Wrote %d clean entries to %s" % (count, dst_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_lmdb_sanity.py /path/to/file.lmdb [--fix /path/to/output.lmdb]")
        sys.exit(1)

    lmdb_path = sys.argv[1]
    bad = check_lmdb(lmdb_path)

    if "--fix" in sys.argv:
        out_path = sys.argv[sys.argv.index("--fix") + 1]
        if bad:
            write_clean_lmdb(lmdb_path, out_path, bad)
        else:
            print("Nothing to fix.")
