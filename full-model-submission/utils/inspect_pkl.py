import pickle

with open("/home/bdldt_team007/DLDockingBenchSeminar/results/proto_test/valid.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Number of batches: {len(data)}")
print(f"Keys in first batch: {list(data[0].keys())}")
b = data[0]
print(f"atoms shape: {b['atoms'].shape}")
print(f"smi_name: {b['smi_name'][:2]}")
print(f"pocket_name: {b['pocket_name'][:2]}")
print(f"coord_predict shape: {b['coord_predict'].shape}")
print(f"holo_coordinates shape: {b['holo_coordinates'].shape}")
print(f"holo_center_coordinates shape: {b['holo_center_coordinates'].shape}")
print(f"prmsd_score shape: {b['prmsd_score'].shape}")
