#!/bin/bash
python3 - << 'PYEOF'
import pickle
import lmdb
import numpy as np
env = lmdb.open("/home/bdldt_team007/DLDockingBenchSeminar/data/processed_posebusters/posebusters_filtered.lmdb",
                subdir=False, readonly=True, lock=False)
txn = env.begin()
data = pickle.loads(txn.get(b"0"))
print("atoms:", data['atoms'][:10])
print("n_atoms:", len(data['atoms']))
print("coord shape:", np.array(data['coordinates'][0]).shape)
print("holo_coord shape:", np.array(data['holo_coordinates'][0]).shape)
PYEOF
