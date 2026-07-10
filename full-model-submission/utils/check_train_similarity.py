import lmdb
import pickle
from collections import Counter

BASE = "/home/bdldt_team007/DLDockingBenchSeminar"

env = lmdb.open(f"{BASE}/data/processed/train.lmdb", subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
txn = env.begin()
keys = sorted(list(txn.cursor().iternext(values=False)), key=lambda k: int(k.decode()))

all_smi = []
all_pockets = []
for k in keys:
    data = pickle.loads(txn.get(k))
    all_smi.append(data['smi'])
    all_pockets.append(data['pocket'])

unique_smi = len(set(all_smi))
print(f"=== TRAIN DATA ===")
print(f"Total complexes: {len(all_smi)}")
print(f"Unique SMILES: {unique_smi}")
print(f"Duplicate ratio: {1 - unique_smi/len(all_smi):.2%}")

pdb_prefixes = [p.split('_')[0] for p in all_pockets]
prefix_counts = Counter(pdb_prefixes)
print(f"\nUnique PDB prefixes (proteins): {len(prefix_counts)}")
print(f"Complexes per protein (top 10): {prefix_counts.most_common(10)}")

# how many SMILES appear only once vs many times
smi_counts = Counter(all_smi)
print(f"\nSMILES appearing only once: {sum(1 for c in smi_counts.values() if c == 1)}")
print(f"SMILES appearing 10+ times: {sum(1 for c in smi_counts.values() if c >= 10)}")
print(f"Most common SMILES (top 5 by count):")
for smi, count in smi_counts.most_common(5):
    print(f"  count={count}: {smi[:50]}")
