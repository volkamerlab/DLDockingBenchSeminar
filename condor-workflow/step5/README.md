# Step5 - Final docking pose prediction (Test-set)

1. ```Testing model```
Yet to produce these results - 10/07/2026
- Repeat step3 and 4 for test dataset using the .sh and .sub files for test.

   The files were manually moved to their respective folders. 
   - Redocked sdf and stat_concated.csv- `dock_results_test/energy_test/ligand_reconstructing`
   - proto_train_final.round0.csv - `data/proto_test_final.round0.csv`

- Run interformer_affinity_pose_final_test.sh via condor_submit interformer_affinity_pose_final_test.sub
- Input: 
   - inference.py via `data/proto_test_final.round0.csv` and checkpoints of the affinity and pose model. Prior to that copy the .sdf files from `dock_results/energy_test/ligand_reconstructing` to `data/proto_test/ligand`(instructed by authors).
- Output:
   - Final csv with predIC50 and pred_pose_score `proto_test_final.round0_ensemble.csv`

2. ```Evaluation```
With the top 1 pose proceed with evaluation.py
