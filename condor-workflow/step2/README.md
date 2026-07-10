# Step 2 - Model training

1. ```ENERGY MODEL TRAINING```

- Run interformer_energy_model.sh via condor_submit interformer_models.sub
- Input:
   - train.py via `data/proto_train_val_final.csv` dataset with hyperparameters(see interformer_energy_model.sh)
- Output: 
   - checkpoints (see checkpoints/energy_model)

2. ```AFFINITY NORMAL MODEL TRAINING```

- Run interformer_affinity_normal_model.sh via condor_submit interformer_models.sub.
- Input:
   - train.py via `data/proto_train_val_final.csv` dataset with hyperparameters(see interformer_affinity_normal_model.sh)
- Output: 
   - checkpoints (see checkpoints/affinity_normal_model)

3. ```AFFINITY AND POSE MODEL TRAINING```

- Implemented after STEP4
- Run interformer_affinity_pose_model.sh via condor_submit interformer_models.sub.
- Input:
   - train.py via updated `data/proto_train_val_final.round0.csv`. Prior to that copy the .sdf files from `dock_results/energy_train/ligand_reconstructing` to `data/proto_train/ligand`(instructed by authors).
- Output:
   - checkpoints (yet to be produced - 10/07/2026)