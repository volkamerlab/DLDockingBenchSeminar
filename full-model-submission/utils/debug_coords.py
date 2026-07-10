import pickle
import lmdb
import numpy as np

BASE = "/home/bdldt_team007/DLDockingBenchSeminar"

# check what the first complex's predicted coords look like before centering
with open(f"{BASE}/results/proto_test/valid.pkl", "rb") as f:
    batches = pickle.load(f)

b = batches[0]
token_mask = b["atoms"][0] > 2
coord_predict = b["coord_predict"][0][token_mask].numpy()
holo_center = b["holo_center_coordinates"][0].numpy()

print(f"coord_predict (first 3 atoms):\n{coord_predict[:3]}")
print(f"holo_center_coordinates (full): {holo_center}")
print(f"holo_center[:3]: {holo_center[:3]}")
print(f"coord_predict + center (first 3):\n{coord_predict[:3] + holo_center[:3]}")
