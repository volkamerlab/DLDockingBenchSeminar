"""
inspect_lmdb.py — quick debugging script to look inside valid.lmdb.

Used this to check what fields each entry actually has, and to confirm
the "pocket" field stores a unique complex_id and not just a bare PDBID
(this is how we caught the duplicate-PDBID naming bug).
"""
import lmdb
import pickle

env = lmdb.open(
    "/home/bdldt_team007/DLDockingBenchSeminar/data/processed/valid.lmdb",
    subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256,
)
txn = env.begin()
keys = list(txn.cursor().iternext(values=False))
print(f"Number of entries: {len(keys)}")

for i in range(5):
    data = pickle.loads(txn.get(keys[i]))
    print(f"--- entry {i} (key={keys[i]}) ---")
    print(f"  keys: {list(data.keys())}")
    print(f"  pocket: {data.get('pocket')}")
    print(f"  smi: {data.get('smi')[:50]}")
