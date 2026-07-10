# Step 1 - Preprocessing

- Manually moved data/proto_train/train_pdb to pocket before running interformer_preprocessing_trainset.sh. The same procedure was carried out for test data set
    - This is because the pocket is technically all of the pdb's -- but the authors had the script set up for a diff file path.

1. ```PRE-PROCESSSING TRAINING and VALIDATION DATASET```

- Run interformer_preprocessing_train.sh and interformer_preprocessing_val.sh via condor_submit interformer_preprocessing.sub.

- Input: 
   - Hamza's folder of full_train and full_val with .pdb and .sdf files
- Output: proto_train folder with:
   - separated .pdb and .sdf files
   - preprocessed pocket (with hydrogens via reduce and extract_pocket_by_ligand.py)
   - preprocessed ligand (with added hydrogens via obabel)
   - energy minimized ligand (uff)

2. ```PRE-PROCESSSING TEST DATASET```

- Run interformer_preprocessing_test.sh via condor_submit interformer_preprocessing.sub.
- Input: 
   - Hamza's folder of full_test with .pdb and .sdf files
- Output: proto_test folder with:
   - separated .pdb and .sdf files (in data/)
   - preprocessed pocket (with hydrogens via reduce and extract_pocket_by_ligand.py)
   - preprocessed ligand (with added hydrogens via obabel)
   - energy minimized ligand (uff)

The proto_train and proto_val was combined into a single csv file using the script/combine_csv.py for easy input for training.