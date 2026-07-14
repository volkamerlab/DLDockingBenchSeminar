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
#9.https://stackoverflow.com/questions/74641071/how-do-i-extract-specific-rows-from-a-csv-file
#10.https://github.com/degrado-lab/PoseBusters-Benchmark

"""
This script does GNINA redocking using the following workflow:
1. Setting the command-line arguments for the test dataset, output and retrained checkpoint directory.
2. Extracting the posebusters dataset CSV and matching each complex to its native ligand and receptor structure files.
4. Running GNINA redocking for each complex using the retrained CNN model.
5. Saving the predicted docked poses as SDF files.

"""
import argparse
import subprocess
from pathlib import Path
import csv
import time


def get_args():
    """
    Parse command-line arguments function for setting parameters for docking, .types file generation, GNINA-Torch training, redocking with checkpoint.
    This function returns argparse.Namespace with including file paths, runtime flags, and configuration settings.
    """
    arg_parser = argparse.ArgumentParser()

    #Posebusters dataset input/output arguments.
    arg_parser.add_argument("--csv", type=Path, default=Path("data/posebusters_filtered.csv"),help="Path to the PoseBusters CSV file")
    arg_parser.add_argument("--input-dir", type=Path, default=Path("data/posebusters_filtered"),help="Directory containing PoseBusters protein and ligand files")
    arg_parser.add_argument("--run", action="store_true", help="Runs GNINA redocking")
    arg_parser.add_argument("--training-out-dir", type=Path, default=Path("results/gninatorch_training/default2018_seed1"), help="Directory where retrained checkpoints and logs are stored")
    arg_parser.add_argument("--retrained-test-out-dir", type=Path, default=Path("results/posebusters_redocked/default2018_seed1"), help="Directory where PoseBusters redocking outputs will be written")

    return arg_parser.parse_args()


def get_files_from_csv_row(csv_row, data_dir: Path):
    """
    This function gets the native ligand and receptor files for dataset CSV row and uses them
    to build the expected file name prefix for a protein-ligand complex. The input arguments are csv_row (dict) that comes from the dataset CSV file including keys like `PDBID`, 
    `Ligand Name`,and input_dir which is the path of the directory contains receptor and ligand structure files. The function returns a tuple containing, complex_id, ligand_path, and eceptor_path.
    """
    ligand_file_name = csv_row.get("ligand_file_name")
    protein_file_name = csv_row.get("protein_file_name")

    if ligand_file_name is None or protein_file_name is None:
        raise RuntimeError(f"Missing required CSV columns in row: {csv_row}")

    ligand_file_name = ligand_file_name.strip() #wanted name prefix of file.
    protein_file_name = protein_file_name.strip()

    ligand_path = data_dir / ligand_file_name
    receptor_path = data_dir / protein_file_name

    if ligand_file_name.endswith("_ligand.sdf"):
        complex_id = ligand_file_name.replace("_ligand.sdf", "")
    else:
        complex_id = Path(ligand_file_name).stem

    if not ligand_path.exists() or not receptor_path.exists():
        raise RuntimeError(
            f"Could not find PoseBusters files for {complex_id}. "
            f"Expected ligand={ligand_path}, protein={receptor_path}"
        )

    return complex_id, ligand_path, receptor_path


def find_latest_checkpoint(training_out_dir: Path):
    """
    The function searches .pt, .pth, .ckpt files to identify most recently modified checkpoint file. (Should be gnina_retrained_full_model.pt)
    It has training_out_dir directory as an input.
    """
    full_model = training_out_dir / "gnina_retrained_full_model.pt"
    if full_model.exists():
        return full_model

    print(f"Expected exported model not found: {full_model}")
    return None


def gnina_args_with_checkpoint(
    receptor_file: Path,
    ligand_file: Path,
    out_file: Path,
    checkpoint_file: Path,
):
    """
    This function generates commands for redocking with a retrained CNN checkpoint.
    It has receptor_file, ligand_file, out_file, and checkpoint_file arguments and it retuns a list for next executions.
    """
    return [
        "gnina", 
        "-r", str(receptor_file), 
        "-l", str(ligand_file),
        "-o", str(out_file),
        "--autobox_ligand", str(ligand_file),
        "--num_modes", "10",
        "--cnn_scoring", "rescore",
        "--cnn_model", str(checkpoint_file),
        "--seed", "1",
    ]


def redock_test_with_retrained_model(
    csv_file: Path,
    input_dir: Path,
    out_dir: Path,
    run: bool,
    training_out_dir: Path,
):
    """
    The function looks for training output directory for the last checkpoint file (retrained CNN). Then, it iterates through the complexes listed in the test CSV including the '--cnn' checkpoint.
    It takes csv_file, nput_dir, out_dir paths, run bool value to decide executing the commands and raining_out_dir path that has retrained model checkpoints.
    """
    checkpoint_file = find_latest_checkpoint(training_out_dir) 
    #error handling in case the checkpoint can't be found:
    if checkpoint_file is None:
        print(f"No checkpoint found in {training_out_dir}.")
        print("Cannot redock proto_test with retrained model.")
        return

    print(f"\nRedocking proto_test with retrained checkpoint: {checkpoint_file}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print(f"Detected headers in {csv_file}: {reader.fieldnames}")

        for csv_row in reader: #parsing through the CSV.
            complex_id, ligand_path, receptor_path = get_files_from_csv_row(csv_row, input_dir)
            if complex_id is None:
                continue
            out_file = out_dir / f"{complex_id}_pred.sdf"

            #Error handling case when there is a missing file.
            if not ligand_path.exists() or not receptor_path.exists():
                print(f"Skipping {complex_id}: missing ligand or receptor")
                continue

            #Skipping docking if an output file already exists
            if out_file.exists():
                print(f"Skipping docking for {complex_id}: found existing output {out_file}")
                continue

            cmd = gnina_args_with_checkpoint(
                receptor_path,
                ligand_path,
                out_file,
                checkpoint_file,
            )

            print("\nRetrained-model docking command:")
            print(" ".join(cmd))

            if not run:
                print("--run flag not set, skipping redocking.")
                continue

            try: #error handling in case redocking fails with model, kill system early to save time.
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Redocking failed for {complex_id}. "
                    f"The exported model was found but is not compatible with GNINA --cnn_model."
                ) from e
            

def main():
    """
    Runs the redocking workflow by using the exported retrained model checkpoint.
    """
    total_start = time.time()
    cli_args = get_args()

    redock_test_with_retrained_model( #redocking over all retrained models.
        cli_args.csv,
        cli_args.input_dir,
        cli_args.retrained_test_out_dir,
        cli_args.run,
        cli_args.training_out_dir,
    )

    print(f"Total runtime: {time.time() - total_start:.1f} seconds")


if __name__ == "__main__":
    main()
