#Irem_Dogruoglu_7061348_Lizzie_Schmitz_7056551
#References:
#1.https://docs.python.org/3/library/argparse.html
#2.https://docs.python.org/3/library/pathlib.html
#3.https://python-adv-web-apps.readthedocs.io/en/latest/csv.html
#4.https://github.com/gnina/gnina
#5.https://link.springer.com/article/10.1186/s13321-025-00973-x
#6.https://docs.python.org/3/library/subprocess.html
#7.https://matplotlib.org/stable/tutorials/introductory/pyplot.html
#8.https://github.com/RMeli/gnina-torch
#9.https://gnina-torch.readthedocs.io/en/latest/api/gninatorch.setup.html#module-gninatorch.setup
#10.https://stackoverflow.com/questions/63967302/pytorch-multi-gpu-issue

"""
This script does a GNINA/GNINA-Torch retraining in the following workflow:

1. Docking the training set with the original GNINA model.
2. Docking the validation set with the original GNINA model.
3. Building GNINA `.types` files for the training and validation sets.
4. Retraining GNINA-Torch models using the training and validation `.types` files.

The script does not redock or evaluate an external test set. These steps will be applied in further separated code files. 
The limitations behind of this were, our runs took so much amount of time regardless how much epochs we were applied. Also, even though we set 2 GPUs in the .sub file, GNINA-Torch primarily operated on a single GPU because the it did not provide guaranteed support for true multi-GPU parallelism.
"""


import argparse
import subprocess
from pathlib import Path
import csv
import time
import re
import matplotlib.pyplot as plt


#Epoch number kept small to reduce runtime.
EPOCHS = 5


def get_args():
    """
    Parse command-line arguments function for setting parameters for docking, .types file generation, GNINA-Torch training.
    This function returns argparse.Namespace with including file paths, runtime flags, and configuration settings.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of GNINA-Torch training iterations")
    arg_parser.add_argument("--training-out-dir", type=Path, default=Path("results/gninatorch_training"), help="Directory where checkpoints and logs are stored")
    arg_parser.add_argument("--training-log", type=Path, default=Path("results/gninatorch_training/training.log"), help="Path to the gnina-torch training log file")

    #Training dataset input/output arguments.
    arg_parser.add_argument("--train-csv", type=Path, default=Path("data/full_sealed_train.csv"))
    arg_parser.add_argument("--train-input-dir", type=Path, default=Path("data/full_sealed_train"))
    arg_parser.add_argument("--train-out-dir", type=Path, default=Path("results/full_sealed_train_redocked"))
    
    #Validation dataset input/output arguments.
    arg_parser.add_argument("--val-csv", type=Path, required=True)
    arg_parser.add_argument("--val-input-dir", type=Path, required=True)
    arg_parser.add_argument("--val-out-dir", type=Path, required=True)

    #Running GNINA docking.
    arg_parser.add_argument("--run", action="store_true", help="Runs GNINA docking")

    #.types file arguments.
    arg_parser.add_argument("--train-types", type=Path, default=Path("data/full_sealed_train.types"))
    arg_parser.add_argument("--val-types", type=Path, required=True)

    arg_parser.add_argument(
        "--make-types",
        action="store_true",
        help="Generated .types files from CSV and results",
    )
    arg_parser.add_argument(
        "--train-model",
        action="store_true",
        help="Runs gnina-torch training",
    )

    #Setting CNN type (Default2018 or Dense) and seed (for reproducibility).
    arg_parser.add_argument(
        "--models",
        nargs="+",
        default=["default2018"],
        help="GNINA-Torch model architectures to train, e.g. default2018 dense",
    )

    arg_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1],
        help="Random seeds for ensemble variants",
    )

    #Adding the plotting arguments.
    arg_parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    arg_parser.add_argument("--make-plots", action="store_true")

    return arg_parser.parse_args()


def get_files_from_csv_row(csv_row, input_dir: Path):
    """
    This function gets the native ligand and receptor files for dataset CSV row and uses them
    to build the expected file name prefix for a protein-ligand complex. The input arguments are csv_row (dict) that comes from the dataset CSV file including keys like `PDBID`, 
    `Ligand Name`,and input_dir which is the path of the directory contains receptor and ligand structure files. The function returns a tuple containing, complex_id, ligand_path, and eceptor_path.
    """
    pdbid = csv_row["PDBID"].strip() 
    ligand_name = csv_row["Ligand Name"].strip()
    chain = csv_row["Ligand Chain"].strip()
    residue = str(csv_row["Ligand Residue Number"]).strip()

    prefix = f"{pdbid}_{ligand_name}_{chain}_{residue}"

    ligand_matches = list(input_dir.glob(f"{prefix}_ligand_refined.sdf"))
    protein_matches = list(input_dir.glob(f"{prefix}_protein_refined.pdb"))

    if not ligand_matches or not protein_matches:
        ligand_matches = list(input_dir.glob(f"{pdbid}_*_{chain}_*_ligand_refined.sdf"))
        protein_matches = list(input_dir.glob(f"{pdbid}_*_{chain}_*_protein_refined.pdb"))

    if len(ligand_matches) != 1 or len(protein_matches) != 1:
        raise RuntimeError(
            f"Could not find one file for PDBID={pdbid}, "
            f"chain={chain}, ligand={ligand_name}, residue={residue}. "
            f"Found ligands={ligand_matches}, proteins={protein_matches}"
        )

    complex_id = ligand_matches[0].name.replace("_ligand_refined.sdf", "")
    return complex_id, ligand_matches[0], protein_matches[0]


def gnina_args(receptor_file: Path, ligand_file: Path, out_file: Path):
    """
    This function gets GNINA docking commands for protein-ligand complexes. 
    Uses arguments as path to the receptor PDB and ligand SDF, and the output file path.
    It returns a list for further execution.
    """
    return [
        "gnina",
        "-r", str(receptor_file),
        "-l", str(ligand_file),
        "-o", str(out_file),
        "--autobox_ligand", str(ligand_file),
        "--num_modes", "10",
        "--cnn_scoring", "rescore",
        "--seed", "1",
    ]


def run_dataset(csv_file: Path, input_dir: Path, out_dir: Path, run: bool, dataset_name: str):
    """
    This function runs docking with checking inputs, skipping missing inputs. It also skips already docked complexes to save time. 
    It uses the CSV file path, the input (input_dir) path, the output (out_dir) path.
    Run to decide which compounds will be docked and to decide dataset label (dataset_name) arguments. 
    """
    start = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset: {dataset_name}")
    print(f"CSV: {csv_file}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {out_dir}")

    with csv_file.open(newline="") as f:
        reader = csv.DictReader(f)

        for csv_row in reader:
            complex_id, ligand_path, receptor_path = get_files_from_csv_row(csv_row, input_dir)
            out_file = out_dir / f"{complex_id}_pred.sdf"

            ligand_miss = not ligand_path.exists()
            receptor_miss = not receptor_path.exists()

            if ligand_miss or receptor_miss:
                print(f"Skipping {complex_id}: missing file(s):")
                if ligand_miss:
                    print(f"Missing ligand: {ligand_path}")
                if receptor_miss:
                    print(f"Missing protein: {receptor_path}")
                continue

            if out_file.exists():
                print(f"Skipping docking for {complex_id}: found existing output {out_file}")
                continue

            cmd = gnina_args(receptor_path, ligand_path, out_file)
            print(
                f"\nComplex {complex_id}\n"
                f"Protein: {receptor_path}\n"
                f"Ligand: {ligand_path}\n"
                f"Output: {out_file}\n"
                f"Command: {' '.join(cmd)}"
            )

            if run:
                subprocess.run(cmd, check=True)

    end = time.time()
    print(f"Finished dataset {dataset_name} in {end - start:.1f} seconds")


def compute_pose_label(native_ligand: Path, predicted_sdf: Path) -> int:
    """
    This function compares the predicted docked pose against native/reference ligand by using 'obrms'. In case of obrms fails, the function returns 1 with a warning so the pipeline can continue.
    It uses path to native ligand file (native_ligand) and predicted GNINA SDF output (predicted_sdf).
    It returns:
    1 for a good pose (RMSD <= 2.0 Å)
    0 for a bad pose (RMSD > 2.0 Å)
    """
    try:
        result = subprocess.run(
            ["obrms", str(native_ligand), str(predicted_sdf)],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()

        if not lines:
            raise ValueError("obrms produced no output")

        first_line = lines[0]
        tokens = first_line.split()

        if len(tokens) < 2: #Adding checks to handle proper input.
            raise ValueError(f"Unexpected obrms output line: {first_line}")

        rmsd = None
        for token in tokens:
            try:
                rmsd = float(token)
                break
            except ValueError:
                continue

        if rmsd is None:
            raise ValueError(f"No numeric RMSD found in line: {first_line}")

        if rmsd <= 2.0:
            return 1
        else:
            return 0

    except FileNotFoundError:
        print("obrms not found in PATH. Labelling 1 for all poses.")
        return 1
    except Exception as e:
        print(f"obrms RMSD computation failed for {predicted_sdf}: {e}")
        return 1


def build_types(csv_file: Path, input_dir: Path, out_dir: Path, types_file: Path):
    """
    This function generates .types file in the expected GNINA format, from the dataset CSV and docking outputs.
    It accepts the csv_file for ligand and protein filenames, input_dir for original receptor and ligand files, out_dir for predicted docking SDF files, and types_file for the output .types file path as inputs.
    """
    print(f"\nBuilding types file: {types_file}")
    print(f"Using CSV: {csv_file}")
    print(f"Input directory: {input_dir}")
    print(f"Results directory: {out_dir}")

    types_file.parent.mkdir(parents=True, exist_ok=True)

    with csv_file.open(newline="") as f_in, types_file.open("w") as f_out:
        reader = csv.DictReader(f_in)

        for csv_row in reader:
            complex_id, native_ligand, receptor_file = get_files_from_csv_row(csv_row, input_dir)
            pred_sdf = out_dir / f"{complex_id}_pred.sdf"

            if not receptor_file.exists() or not pred_sdf.exists() or not native_ligand.exists():
                print(f"Skipping {complex_id} in types: missing receptor, predicted SDF, or native ligand")
                continue

            label = compute_pose_label(native_ligand, pred_sdf)
            affinity = -float(csv_row["Log Binding Affinity"])

            f_out.write(f"{label} {affinity} {receptor_file} {pred_sdf}\n")


def run_training(train_types: Path, val_types: Path, EPOCHS: int, training_out_dir: Path, training_log: Path, model_name: str, seed: int):
    """
    This function calls GNINA-Torch training as a subprocess. Then the output is written into the output directory.
    It takes train_types as the path to the training file, val_types as the path to the validation file, EPOCHS as the number of training iterations, training_out_dir as the directory for saving checkpoints and other outputs, and training_log as where the training log is written.
    """
    training_out_dir.mkdir(parents=True, exist_ok=True)
    training_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m", "gninatorch.training",
        str(train_types),
        "--model", model_name,
        "--seed", str(seed),
        "--affinity_pos", "1",
        "--batch_size", "32",
        "--iterations", str(EPOCHS),
        "--test_every", "1",
        "--checkpoint_every", str(EPOCHS),
        "--num_checkpoints", "1",
        "--out_dir", str(training_out_dir),
    ]

    if val_types is not None:
        cmd.extend(["--testfile", str(val_types)])

    print("\nRunning gnina-torch training with command:")
    print(" ".join(cmd))

    try:
        start = time.time()

        with training_log.open("w") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"gnina-torch training failed with exit code {proc.returncode}. "
                f"Check log: {training_log}"
            )
        else:
            total_hours = (time.time() - start) / 3600.0
            print(f"Training completed in {total_hours:.2f} hours.")
            print(f"Training log saved to: {training_log}")

    except FileNotFoundError:
        print("gninatorch was not found.")
        print("training step is skipped, check gnina-torch installation.")
    except Exception as e:
        print(f"Exception while running gnina-torch training: {e}")
        print("Skipped training step, check the error.")


def find_latest_checkpoint(training_out_dir: Path):
    """
    The function searches .pt, .pth, .ckpt files to identify most recently modified checkpoint file. (Should be gnina_retrained_full_model.pt)
    It has training_out_dir directory as an input.
    """
    full_model = training_out_dir / "gnina_retrained_full_model.pt"
    if full_model.exists():
        return full_model

    raise FileNotFoundError(
        f"Expected exported model not found: {full_model}"
    )


def parse_gnina_sdf_scores(sdf_file: Path):
    """
    This function reads the full SDF text and searches for multiple possible name variants, so that GNINA naming conventions can still be visible.
    Uses argument of sdf_file as SDF output file, and returns a tuple containing all parsed CNN pose score and CNN affinity score values.
    """
    text = sdf_file.read_text(errors="ignore")

    pose_scores = []
    affinity_scores = []

    score_patterns = ["CNNscore", "CNN_score", "CNN_pose_score", "CNN Pose Score", "CNN pose score"]
    affinity_patterns = ["CNNaffinity", "CNN_affinity", "CNN Affinity", "CNN affinity"]

    for pattern in score_patterns:
        matches = re.findall(rf">\s*<\s*{re.escape(pattern)}\s*>\s*\n([-\d.eE]+)", text)
        pose_scores.extend(float(x) for x in matches)

    for pattern in affinity_patterns:
        matches = re.findall(rf">\s*<\s*{re.escape(pattern)}\s*>\s*\n([-\d.eE]+)", text)
        affinity_scores.extend(float(x) for x in matches)

    return pose_scores, affinity_scores


def compute_rmsd_value(native_ligand: Path, predicted_sdf: Path):
    """
    This function calculates an RMSD value between a native ligand and predicted docked pose.
    It uses native_ligand and he predicted GNINA SDF file (predicted_sdf) paths, returning float of RMSD value.
    """
    try:
        result = subprocess.run(
            ["obrms", str(native_ligand), str(predicted_sdf)],
            check=True,
            capture_output=True,
            text=True,
        )

        for line in result.stdout.strip().splitlines():
            tokens = line.split()
            for token in tokens:
                try:
                    return float(token)
                except ValueError:
                    continue

    except Exception as e:
        print(f"RMSD failed for {predicted_sdf}: {e}")

    return None


def main():
    """
    Runs the main GNINA workflow by including docking on training datasets, .types file generation, and GNINA-Torch training.
    """
    total_start = time.time()

    cli_args = get_args()

    trained_model_dirs = []

    #Step 1: Docking the training and validation sets with GNINA.
    run_dataset(cli_args.train_csv, cli_args.train_input_dir, cli_args.train_out_dir, cli_args.run, "TRAIN DATA")
    run_dataset(cli_args.val_csv, cli_args.val_input_dir, cli_args.val_out_dir, cli_args.run, "VALIDATION DATA")

    #Step 2: Building train and validation .types files.
    if cli_args.make_types:
        t0 = time.time()

        build_types(
            cli_args.train_csv,
            cli_args.train_input_dir,
            cli_args.train_out_dir,
            cli_args.train_types,
        )

        build_types(
            cli_args.val_csv,
            cli_args.val_input_dir,
            cli_args.val_out_dir,
            cli_args.val_types,
        )

        print(f"Building train/val .types files took {time.time() - t0:.1f} seconds")
    else:
        print("\nSkipping .types preprocessing.")

    #Step 3: Training GNINA-Torch model variants.
    if cli_args.train_model:
        t1 = time.time()
        for model_name in cli_args.models:
            for seed in cli_args.seeds:
                run_name = f"{model_name}_seed{seed}"
                model_out_dir = cli_args.training_out_dir / run_name
                model_log = model_out_dir / "training.log"

                run_training(
                    cli_args.train_types,
                    cli_args.val_types,
                    cli_args.epochs,
                    model_out_dir,
                    model_log,
                    model_name,
                    seed,
                )
                checkpoint_file = find_latest_checkpoint(model_out_dir)
                print(f"submitting the checkpoint: {checkpoint_file}")
                trained_model_dirs.append(model_out_dir)

        print(f"Training step took {time.time() - t1:.1f} seconds")
    else:
        print("Skipping model training.")

    print(f"Total runtime: {time.time() - total_start:.1f} seconds")


if __name__ == "__main__":
    main()
