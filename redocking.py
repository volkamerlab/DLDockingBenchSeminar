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

    #Test dataset input/output arguments.
    arg_parser.add_argument("--csv", type=Path, default=Path("data/full_sealed_test.csv"))
    arg_parser.add_argument("--input-dir", type=Path, default=Path("data/full_sealed_test"))
    arg_parser.add_argument("--run", action="store_true", help="Runs GNINA docking")

    arg_parser.add_argument(
        "--training-out-dir",
        type=Path,
        default=Path("results/gninatorch_training/default2018_seed1"),
        help="Directory where checkpoints and logs are stored",
    )

    arg_parser.add_argument(
        "--retrained-test-out-dir",
        type=Path,
        default=Path("results/full_sealed_test_retrained/default2018_seed1"),
        help="Creates output directory",
    )

    return arg_parser.parse_args()



def get_files_from_csv_row(csv_row, input_dir: Path): # parse through "3ix2_AC2_A_302_ligand_refined.sdf" format
    pdbid = csv_row["PDBID"].strip() 
    ligand_name = csv_row["Ligand Name"].strip()
    chain = csv_row["Ligand Chain"].strip()
    residue = str(csv_row["Ligand Residue Number"]).strip()

    prefix = f"{pdbid}_{ligand_name}_{chain}_{residue}" # name prefix of file

    # glob finds all files in the dir that match the prefix name
    ligand_matches = list(input_dir.glob(f"{prefix}_ligand_refined.sdf"))
    protein_matches = list(input_dir.glob(f"{prefix}_protein_refined.pdb"))

    if not ligand_matches or not protein_matches: # error handling case, if no files with prefix are found in the dir 
        ligand_matches = list(input_dir.glob(f"{pdbid}_*_{chain}_*_ligand_refined.sdf"))
        protein_matches = list(input_dir.glob(f"{pdbid}_*_{chain}_*_protein_refined.pdb"))

    if len(ligand_matches) != 1 or len(protein_matches) != 1: # error handling: 0 or >1 files with name found
        raise RuntimeError(
            f"Could not find one file for PDBID={pdbid}, "
            f"chain={chain}, ligand={ligand_name}, residue={residue}. "
            f"Found ligands={ligand_matches}, proteins={protein_matches}"
        )

    complex_id = ligand_matches[0].name.replace("_ligand_refined.sdf", "")
    return complex_id, ligand_matches[0], protein_matches[0] 


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
    # error handling: in case the checkpoint can't be found:
    if checkpoint_file is None:
        print(f"No checkpoint found in {training_out_dir}.")
        print("Cannot redock proto_test with retrained model.")
        return


    print(f"\nRedocking proto_test with retrained checkpoint: {checkpoint_file}")


    out_dir.mkdir(parents=True, exist_ok=True)


    with csv_file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print(f"Detected headers in {csv_file}: {reader.fieldnames}")


        for csv_row in reader: # parse through the CSV
            complex_id, ligand_path, receptor_path = get_files_from_csv_row(csv_row, input_dir)
            if complex_id is None:
                continue
            out_file = out_dir / f"{complex_id}_pred.sdf"

            # error handling case:
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
            # error handling: 
            if not run:
                print("--run flag not set, skipping redocking.")
                continue


            try: # error handling: in case redocking fails with model, kill system early to save time
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


    redock_test_with_retrained_model(
        cli_args.csv,
        cli_args.input_dir,
        cli_args.retrained_test_out_dir,
        cli_args.run,
        cli_args.training_out_dir,
    )


    print(f"Total runtime: {time.time() - total_start:.1f} seconds")



if __name__ == "__main__":
    main()
