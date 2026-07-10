import lmdb, pickle
from collections import Counter

env = lmdb.open(
    "/home/bdldt_team007/DLDockingBenchSeminar/data/processed/train.lmdb",
    subdir=False, readonly=True, lock=False
)
all_atoms = Counter()
with env.begin() as txn:
    for i in range(20):
        entry = pickle.loads(txn.get(f"{i}".encode()))
        for a in entry['pocket_atoms']:
            all_atoms[a[0]] += 1

print("Pocket atom types found:", dict(all_atoms))
