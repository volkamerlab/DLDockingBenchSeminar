# Interformer
--------------------
## Benchmarking DL-Based Docking Tools

Team Members: Lakshana & Ben

Supervisors: Hamza Ibrahim & Andrea Volkamer

-----------
Final Submission Notice: We have the final poses being uploaded to zenodo currently, as well as the checkpoints being run right now (intermediate checkpoint with 5 epochs uploaded), we will upload them ASAP, thank you for your patience! :)
-----------


### Notes re: File Structure
-----------
1. Renamed full_data/ to data/ to reduce reproducibility steps. Also 
2. Manually separated .pdb and .sdf files from  into:
   - /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train/train_pdb
   - /home/bdldt_team001/DLDockingBenchSeminar/data/proto_train/train_sdf
3. Check README.md files under each steps in condor_workflow/ to understand the workflow of the training
4. Poses predicted by the model - results/
5. Loss curve of energy and affinity normal model - loss_curve/ (Affinity pose not included as trained only for 5 epochs)

--------------------

## Overview of Steps
Please note that more details of each step can be found in the README of each step. 

# Interformer Pipeline Summary

| Step | Purpose | Input / Output |
|---|---|---|
| **1: Preprocessing** | Prepares raw protein/ligand data for training: adds hydrogens to ligands (obabel), generates initial 3D conformations (UFF via `rdkit_ETKDG_3d_gen.py`), cleans protein structures (Reduce) to avoid steric clashes, and extracts the binding pocket within 10 Å of the reference ligand (`extract_pocket_by_ligand.py`). Train/val/test sets are each processed with a dedicated script, then combined into one CSV (`combine_csv.py`). | **Input:**<br>Raw folders of `.pdb`/`.sdf` files (full_train/full_val/full_test).<br><br>**Output:**<br>`proto_train`/`proto_val`/`proto_test` folders (separated pdb/sdf, processed pocket, ligand, and UFF-optimized ligand); merged `proto_train_val_final.csv` for training. |
| **2: Model training** | Trains three Interformer model variants using `train.py`: <br> (a) **Energy model** — baseline pose-scoring model; <br>(b) **Affinity normal model** — predicts affinity from original (non-augmented) poses only <br>(c) **Affinity+pose model** — trained on positive/negative pose samples generated via redocking with the energy-model checkpoints (implemented after Step 4). | **Input:**<br>`proto_train_val_final.csv` (and later `proto_train_val_final.round0.csv` for the pose model, after copying redocked `.sdf`s from Step 4 into `data/proto_train/ligand`).<br><br>**Output:**<br>Model checkpoints (`checkpoints/energy_model`, `checkpoints/affinity_normal_model`, `checkpoints/affinity_pose_model`). |
| **3: Gaussian function & complex generation** | Runs inference with the trained energy-model checkpoints to predict Gaussian interaction scores and generate protein-ligand complex structures needed for the subsequent docking/negative-sampling step. | **Input:**<br>`proto_train_val_final.csv` + `checkpoints/energy_model`.<br><br>**Output:**<br>Gaussian interaction predictions and generated complex structures (feeds into Step 4's docking folder). |
| **4: Docking poses generation** | Reconstructs ligand poses from the Step 3 outputs, generates docking statistics (RMSD, pose rank, van der Waals distance, energy), and merges these stats back into the training CSV. Poses with RMSD ≤ cutoff (default 2 Å) are kept as positive samples poses above the cutoff are relabeled with negative pIC50 via `label_negatives.py` to create contrastive positive/negative training data for the affinity+pose model. | **Input:**<br>Complex/Gaussian/ligand/UFF outputs from Step 3 (unpacked from tarballs).<br><br>**Output:**<br>Redocked `.sdf`'s (`ligand_reconstructing`), `stat_concated.csv`, and the labeled `proto_train_val_final.round0.csv` (with signed pIC50 for pos/neg samples). |
| **5: Final docking pose prediction (test set)** | Repeats Steps 3–4 on the held-out test set, then runs final inference using the trained affinity+pose model checkpoints to score docking poses and predict pIC50/pose scores for the test ligands. | **Input:**<br>`proto_test_final.round0.csv` + affinity+pose model checkpoints (redocked test `.sdf`s copied into `data/proto_test/ligand`).<br><br>**Output:**<br>Final scored CSV `proto_test_final.round0_ensemble.csv` (predIC50, pred_pose_score); <br>top-1 pose used downstream in `evaluation.py`. |



Results
----
Plots can be found at: https://wandb.ai/dl-docking/Interformer?nw=nwuserbeho00003 (shared with Hamza)


## Conclusion

We have learned a lot from this experience to experiment and reproduce results for Interformer and would like to thank Revo(Interformer author) and Hamza for the mentorship; as well as Professor Volkamer the opportunity to partake in this seminar!
