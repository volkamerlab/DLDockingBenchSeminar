--- [ 1. DATA PREPARATION & CLEANING ] ----------------------------------------

create_target_train.py / create_target_test.py
    Builds a standardized "Target" identifier column (PDBID_Ligand_Chain_Residue 
    for train/val; parsed from the ligand filename for test) and inserts it as 
    the first column so downstream Interformer scripts can key on a single 
    complex ID.

rename_docked_sdf_train.py / rename_docked_sdf_test.py
    Batch-renames OpenBabel output ligand files from the "*_ligand_refined.sdf" 
    suffix to the "*_docked.sdf" suffix required by Interformer's downstream 
    docking/pipeline scripts.

combine_csv.py
    Concatenates the finalized prototype train and validation CSVs into a single 
    combined train+val CSV for full-dataset model training.


--- [ 2. FEATURE & ANNOTATION EXTRACTION ] ------------------------------------

get_IC50_uniprot_train.py / get_IC50_uniprot_test.py
    Computes pIC50 (-log Binding Affinity) for each entry in the prototype 
    train/test CSVs and cross-references PDB IDs against the RCSB GraphQL API 
    (batched) to append missing UniProt IDs, replicating a feature present in 
    the original dataset but absent from the prototype data.

rdkit_ETKDG_3d_gen.py
    Generates ligand conformers using RDKit's ETKDGv3 algorithm (30 embeddings), 
    UFF-optimizes each, and saves the lowest-energy conformer as the _uff.sdf 
    reference structure, reporting energy and RMSD vs. the original input.

extract_pocket_by_ligand.py
    Extracts the protein binding pocket (10 Angstrom cutoff) around a reference 
    ligand using ODDT's ExtractPocketAndLigand, identifying and optionally 
    excluding the ligand's own CCD code so other bound heteroatoms (metals, 
    cofactors) are preserved in the output pocket PDB.


--- [ 3. MODEL TRAINING & INFERENCE ] -----------------------------------------

train.py
    Main PyTorch Lightning training entry point for Interformer (energy, pose, 
    or affinity-normal models); configures the data module, model, and Weights 
    & Biases logging, then launches distributed (DDP) training via the Trainer.

inference.py
    Runs multi-GPU (DDP) inference across an ensemble of trained Interformer 
    checkpoints to predict pIC50/pose scores per target, then selects the top-8 
    performing models and blends their outputs into a consensus ensemble CSV 
    with prediction variance and Pearson correlation metrics.


--- [ 4. DOCKING RECONSTRUCTION & POST-PROCESSING ] ---------------------------

reconstruct_ligands.py
    CLI-driven docking reconstruction tool that runs Monte Carlo pose sampling 
    (64 repeats x 2000 steps) per target using predicted energy grids--either 
    locally or via remote/cluster dispatch. Generates docked ligand poses, 
    compiles a detailed docking summary (RMSD, energy, torsions, pose rank), 
    and aggregates per-target results into a summary CSV reporting RMSD pass-rate 
    and steric-clash ("bust") pass-rate across all docked targets.
    [Note: Inferred from big-README.md usage - source file not provided]

merge_summary_input.py
    Merges the docking summary statistics (pose_rank, num_torsions, energy, rmsd) 
    produced by the reconstruct_ligands.py pipeline back into the main input 
    datasets.

===============================================================================