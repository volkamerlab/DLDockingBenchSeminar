# Step4 - Docking poses generation

1. ```Docking poses generation```
- Run interformer_affinity_pose_docking.sh via interformer_affinity_pose_docking.sub
- Input: 
   - Gaussian predictions of interactions and complex of protein structures with ligand and uff from `data/proto_train/`  in the dock_results/energy_train
- Output: 
   - Statistics on reconstructed ligand poses and generation of updated csv file (Informations included: rmsd, pose_rank, vdw_distance, energy).


Yet to produce these files (10/07/2026)
   The files were manually moved to their respective folders. 
   - Redocked sdf - `dock_results/energy_train/ligand_reconstructing`
   - proto_train_val_final.round0.csv - `data/proto_train_val_final.round0.csv`
   - stat_concated.csv - `dock_results/energy_train/stat_concated.csv`

2. ```Negative samples labelling```

- *Note on Generating Negative Poses: After generating the proto_train_val_final.csv file, we confirmed with the author that in order to generate the negative poses of having a RMSD value >2A, we just adjust the column (pIC50) to become negative.