import lmdb, pickle

for path, name in [
    ("/home/bdldt_team007/DLDockingBenchSeminar/data/processed/train.lmdb", "train"),
    ("/home/bdldt_team007/DLDockingBenchSeminar/data/processed/valid.lmdb", "valid"),
]:
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        length = env.stat()['entries']
        first = pickle.loads(txn.get(b"0"))
    print(f"{name}: {length} samples")
    print(f"  keys in entry: {list(first.keys())}")
    print(f"  atoms[:5]: {first['atoms'][:5]}")
    print(f"  num conformers: {len(first['coordinates'])}")
    print(f"  pocket size: {len(first['pocket_atoms'])}")
