import torch
import numpy as np

for name, path in [
    ("mol", "/home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/weights/mol_pre_no_h_220816.pt"),
    ("pocket", "/home/bdldt_team007/DLDockingBenchSeminar/unimol/unimol_docking_v2/weights/pocket_pre_220816.pt"),
]:
    state = torch.load(path, map_location='cpu')
    model = state['model']
    nan_keys = [k for k, v in model.items() if torch.isnan(v).any()]
    print(f"\n{name} weights: {len(model)} tensors")
    print(f"  NaN tensors: {len(nan_keys)}")
    if nan_keys:
        print(f"  First NaN key: {nan_keys[0]}")
