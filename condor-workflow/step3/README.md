# Step3 - Gaussian function and complex generation

1. ```Gaussian function and complex generation```
- Run interformer_affinity_pose_train_data.sh via condor_submit interformer_affinity_pose_train_data.sub for gaussian score prediction and complex generation.
- Input: 
   - inference.py via `data/proto_train_val_final.csv` and checkpoints of the energy model
- Output: 
   - Gaussian predictions of interactions and complex of protein structures.