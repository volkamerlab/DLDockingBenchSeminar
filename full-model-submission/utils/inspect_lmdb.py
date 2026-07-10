import lmdb
import pickle

env = lmdb.open(
    "/home/bdldt_team007/DLDockingBenchSeminar/data/processed/valid.lmdb",
    subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256,
)
txn = env.begin()
keys = list(txn.cursor().iternext(values=False))
print(f"Number of entries: {len(keys)}")

pockets = []
for k in keys:
    data = pickle.loads(txn.get(k))
    pockets.append(data['pocket'])

print(f"Unique pockets: {len(set(pockets))}")
print(f"First 10: {pockets[:10]}")
