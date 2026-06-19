"""
inspect_pkl.py — quick debugging script to see what's actually inside valid.pkl.

We used this once, early on, just to figure out what fields infer.py
actually saves (since the official docs don't spell it out clearly) before
writing pkl_to_sdf.py.
"""
import pickle

with open("/home/bdldt_team007/DLDockingBenchSeminar/results/proto_test/valid.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Number of batches: {len(data)}")
print(f"Keys in first batch: {list(data[0].keys())}")

b = data[0]
print(f"atoms shape: {b['atoms'].shape}")                                   # token ids, includes padding
print(f"smi_name: {b['smi_name'][:2]}")                                     # SMILES strings
print(f"pocket_name: {b['pocket_name'][:2]}")                               # complex_id (despite the name)
print(f"coord_predict shape: {b['coord_predict'].shape}")                   # predicted xyz coordinates
print(f"holo_coordinates shape: {b['holo_coordinates'].shape}")             # ground truth coordinates
print(f"holo_center_coordinates shape: {b['holo_center_coordinates'].shape}")  # offset to add back later
print(f"prmsd_score shape: {b['prmsd_score'].shape}")                       # model's own confidence score
