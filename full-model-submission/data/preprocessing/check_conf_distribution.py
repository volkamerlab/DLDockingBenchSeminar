import sys
import pickle
import lmdb
from collections import Counter


def check_conf_distribution(lmdb_path, expected_conf_size=10):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    n_total = 0
    conf_count_dist = Counter()
    bad_entries = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            n_total += 1
            entry = pickle.loads(value)
            coords = entry.get("coordinates", [])
            n_confs = len(coords) if coords is not None else 0
            conf_count_dist[n_confs] += 1

            if n_confs < expected_conf_size:
                bad_entries.append((key.decode(), n_confs, entry.get("pocket", "?")))

    env.close()

    print("")
    print("=== %s ===" % lmdb_path)
    print("Total entries: %d" % n_total)
    print("")
    print("Conformer count distribution (n_conformers -> n_entries):")
    for n_confs in sorted(conf_count_dist.keys()):
        print("  %2d conformers -> %5d entries" % (n_confs, conf_count_dist[n_confs]))

    print("")
    print("Entries with FEWER than %d conformers (would crash TTADockingPoseDataset with --conf-size %d):" % (expected_conf_size, expected_conf_size))
    print("  Count: %d / %d (%.2f%%)" % (len(bad_entries), n_total, 100.0 * len(bad_entries) / n_total if n_total else 0))

    if bad_entries:
        print("")
        print("First 20 problem entries (key, n_conformers, pocket/pdbid):")
        for k, n_confs, pocket in bad_entries[:20]:
            print("  key=%s  n_conformers=%d  pocket=%s" % (k, n_confs, pocket))
        if len(bad_entries) > 20:
            print("  ... and %d more" % (len(bad_entries) - 20))

    return bad_entries


def write_dropped_lmdb(src_path, dst_path, bad_entry_keys):
    bad_key_set = set(bad_entry_keys)
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
    print("Wrote %d entries (dropped %d) to %s" % (count, len(bad_key_set), dst_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_conf_distribution.py /path/to/valid.lmdb [expected_conf_size] [--drop /path/to/output.lmdb]")
        sys.exit(1)

    lmdb_path = sys.argv[1]
    expected_conf_size = 10
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        expected_conf_size = int(sys.argv[2])

    bad = check_conf_distribution(lmdb_path, expected_conf_size=expected_conf_size)

    if "--drop" in sys.argv:
        out_path = sys.argv[sys.argv.index("--drop") + 1]
        if bad:
            bad_keys = [k for k, _, _ in bad]
            write_dropped_lmdb(lmdb_path, out_path, bad_keys)
        else:
            print("Nothing to drop.")
