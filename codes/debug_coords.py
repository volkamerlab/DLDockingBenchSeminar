"""
debug_coords.py — checked the raw numbers behind our collapsed-pose problem.

When we noticed all atoms in our predicted SDF files were landing on
basically the same point, we used this to look at the actual coord_predict
and holo_center_coordinates values straight out of valid.pkl, before any
of our own postprocessing touched them — just to rule out a bug in
pkl_to_sdf.py itself before concluding it was a model training issue.
"""
import pickle
import numpy as np

BASE = "/home/bdldt_team007/DLDockingBenchSeminar"

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
