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

"""
This script automates a workflow for GNINA and GNINA-Torch by following steps:

1. Using GNINA to dock the proto_train dataset.
2. Creating .types files from docking outputs and retraining GNINA-Torch model(s) (dense or default2018) on the new training dataset
3. After retraining, using the new checkpoint (.pt) files to re-dock proto_test.
4. Obtaining training-loss metrics, CNN posture score, CNN affinity, and RMSD from the checkpoints.
5. Generating plots and CSV summaries for further analysis.
"""

import argparse
import subprocess
from pathlib import Path
import csv
import time
import re
import matplotlib.pyplot as plt

#global var
#setting the epoch to 5 to observe curves and handle data better.
EPOCHS = 5

def get_args():
    """
    Parse command-line arguments function for setting parameters for docking, .types file generation, GNINA-Torch training, redocking with checkpoint.
    This function returns argparse.Namespace with including file paths, runtime flags, and configuration settings.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of GNINA-Torch training iterations")
    arg_parser.add_argument("--training-out-dir", type=Path, default=Path("results/gninatorch_training"), help="Directory where checkpoints and logs are stored")
    arg_parser.add_argument("--training-log", type=Path, default=Path("results/gninatorch_training/training.log"), help="Path to the gnina-torch training log file")

    #Training dataset input/output arguments
    arg_parser.add_argument("--train-csv", type=Path, default=Path("data/proto_train.csv"))
    arg_parser.add_argument("--train-input-dir", type=Path, default=Path("data/proto_train"))
    arg_parser.add_argument("--train-out-dir", type=Path, default=Path("results/proto_train"))
    
    # LATER: Validation dataset input/output arguments
   # arg_parser.add_argument("--val-csv", type=Path, default=Path("data/validation.csv"))
   # arg_parser.add_argument("--val-input-dir", type=Path, default=Path("validation"))
   # arg_parser.add_argument("--val-types", type=Path, default=Path("data/validation.types"))

    #Test dataset input/output arguments.
    arg_parser.add_argument("--csv", type=Path, default=Path("data/proto_test.csv"))
    arg_parser.add_argument("--input-dir", type=Path, default=Path("data/proto_test"))
    arg_parser.add_argument("--run", action="store_true", help="Runs GNINA docking")

    #GNINA/gnina-torch .types file arguments.
    arg_parser.add_argument("--train-types", type=Path, default=Path("data/proto_train.types"))

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
    arg_parser.add_argument(
        "--retrained-test-out-dir",
        type=Path,
        default=Path("results/proto_test_retrained"),
        help="Creates output directory",
    )

    # CNN type (Default2018 or Dense) and seed (for reproducibility)
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


    # Plotting args
    arg_parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    arg_parser.add_argument("--make-plots", action="store_true")

    return arg_parser.parse_args()


def get_comp_id(ligand_file: str) -> str:
    """
    The function extracts the complex ID from a ligand filename, by using ligand name from the dataset CSV. 
    It returns complex ID in str form.
    """
    if ligand_file.endswith("_ligand_refined.sdf"):
        comp_id = ligand_file.replace("_ligand_refined.sdf", "")
    elif ligand_file.endswith("_ligand.sdf"):
        comp_id = ligand_file.replace("_ligand.sdf", "")
    else:
        comp_id = Path(ligand_file).stem
    return comp_id


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
            ligand_file = csv_row["ligand_file_name"]
            protein_file = csv_row["protein_file_name"]

            #Generating ID from the ligand filename
            complex_id = get_comp_id(ligand_file)

            #Building input and output paths
            ligand_path = input_dir / ligand_file
            receptor_path = input_dir / protein_file
            out_file = out_dir / f"{complex_id}_pred.sdf"

            #Skipping complexes with missing source files
            ligand_miss = not ligand_path.exists()
            receptor_miss = not receptor_path.exists()

            if ligand_miss or receptor_miss:
                print(f"Skipping {complex_id}: missing file(s):")
                if ligand_miss:
                    print(f"Missing ligand: {ligand_path}")
                if receptor_miss:
                    print(f"Missing protein: {receptor_path}")
                continue

            #Skipping docking if an output file already exists
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
            ["obrms", str(native_ligand), str(predicted_sdf)], # obrms check function
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()

        # Error handling cases
        if not lines:
            raise ValueError("obrms produced no output")

        first_line = lines[0]
        tokens = first_line.split()

        if len(tokens) < 2:
            raise ValueError(f"Unexpected obrms output line: {first_line}")

        # extract RMSD values
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
            ligand_name = csv_row["ligand_file_name"]
            protein_name = csv_row["protein_file_name"]

            complex_id = get_comp_id(ligand_name)

            receptor_file = input_dir / protein_name
            pred_sdf = out_dir / f"{complex_id}_pred.sdf" 
            native_ligand = input_dir / ligand_name

            # error handling for missing files, skip
            if not receptor_file.exists() or not pred_sdf.exists() or not native_ligand.exists():
                print(f"Skipping {complex_id} in types: missing receptor, predicted SDF, or native ligand")
                continue

            # get pose labels and affinities from CSV
            label = compute_pose_label(native_ligand, pred_sdf)
            affinity = -float(csv_row["Log Binding Affinity"]) # convert to positive pK affinity vals 
            # previously, most of the affinity labels were all negative log values (in kcal/mol), derived by log10(Kd), according to AutoDock VINA
            # GNINA expects pK vals, s.t. pK = -log10(Kd), so we just multiply by -1
            
            # computing GNINA-style output line format in label-affinity-receptor-ligand order
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
        "--model", model_name, # default2018 or dense
        "--seed", str(seed),
        "--affinity_pos", "1",
        "--batch_size", "32",
        "--iterations", str(EPOCHS), # (EPOCHS) number of epochs
        "--test_every", "1",
        "--checkpoint_every", str(EPOCHS), # train every (EPOCH) epochs
        "--num_checkpoints", "1",
        "--out_dir", str(training_out_dir),
    ]

    # LATER: we need to use a validation set, add it here 
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

         # continuing pipeline execution even if training fails
        if proc.returncode != 0:
            print(f"gnina-torch training failed with exit code {proc.returncode}.")
        else: # timing reports
            total_hours = (time.time() - start) / 3600.0
            print(f"Training completed in {total_hours:.2f} hours.")
            print(f"Training log saved to: {training_log}")
    # error handling
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
    full_model = training_out_dir / "gnina_retrained_full_model.pt" # as defined in the dockerfile, to be output
    if full_model.exists():
        return full_model

    raise FileNotFoundError(
    f"Expected exported model not found: {full_model}"
    )

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

    with csv_file.open(newline="") as f:
        reader = csv.DictReader(f)

        for csv_row in reader: # parse through the CSV
            ligand_file = csv_row["ligand_file_name"]
            protein_file = csv_row["protein_file_name"]

            complex_id = get_comp_id(ligand_file)

            ligand_path = input_dir / ligand_file
            receptor_path = input_dir / protein_file
            out_file = out_dir / f"{complex_id}_pred.sdf"
            # error handling case:
            if not ligand_path.exists() or not receptor_path.exists():
                print(f"Skipping {complex_id}: missing ligand or receptor")
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



def parse_gnina_sdf_scores(sdf_file: Path):
    """
    This function reads the full SDF text and searches for multiple possible name variants, so that GNINA naming conventions can still be visible.
    Uses argument of sdf_file as SDF output file, and returns a tuple containing all parsed CNN pose score and CNN affinity score values.
    """
    text = sdf_file.read_text(errors="ignore")

    pose_scores = []
    affinity_scores = []

    #Including multiple labels used in GNINA SDF files for flexibility
    score_patterns = ["CNNscore", "CNN_score", "CNN_pose_score", "CNN Pose Score", "CNN pose score"]
    affinity_patterns = ["CNNaffinity", "CNN_affinity", "CNN Affinity", "CNN affinity"]

    #Extracting floating-point values in SDF property tags
    # Use regex expressions for more flexibility
    for pattern in score_patterns:
        matches = re.findall(rf">\s*<\s*{re.escape(pattern)}\s*>\s*\n([-\d.eE]+)", text)
        pose_scores.extend(float(x) for x in matches)

    for pattern in affinity_patterns:
        matches = re.findall(rf">\s*<\s*{re.escape(pattern)}\s*>\s*\n([-\d.eE]+)", text)
        affinity_scores.extend(float(x) for x in matches)

    return pose_scores, affinity_scores


def compute_rmsd_value(native_ligand: Path, predicted_sdf: Path): # replace with evaluation.py script later, after prototype submisison
    """
    This function calculates an RMSD value between a native ligand and predicted docked pose.
    It uses native_ligand and he predicted GNINA SDF file (predicted_sdf) paths, returning float of RMSD value.
    """
    try:
        result = subprocess.run( # same obrms function from compute_pose_label()
            ["obrms", str(native_ligand), str(predicted_sdf)],
            check=True,
            capture_output=True,
            text=True,
        )
        # parse through results
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


def collect_metrics(csv_file: Path, input_dir: Path, out_dir: Path, dataset_name: str):
    """
    This function gathers all RMSD, CNN pose scores, and CNN affinity scores for the dataset. Then results are stored as a list of dictionaries.
    It takes csv_file as ligand filenames for the dataset, input_dir, out_dir paths, and dataset_name as the label stored in the output rows.
    """
    rows = []

    with csv_file.open(newline="") as f:
        reader = csv.DictReader(f)

        for csv_row in reader:
            ligand_name = csv_row["ligand_file_name"]
            complex_id = get_comp_id(ligand_name)

            native_ligand = input_dir / ligand_name
            pred_sdf = out_dir / f"{complex_id}_pred.sdf"

            if not pred_sdf.exists():
                continue

            pose_scores, affinity_scores = parse_gnina_sdf_scores(pred_sdf)
            rmsd = compute_rmsd_value(native_ligand, pred_sdf)

            rows.append( # dict structure
                {
                    "dataset": dataset_name,
                    "complex_id": complex_id,
                    "rmsd": rmsd,
                    "best_cnn_pose_score": pose_scores[0] if pose_scores else None, # best pose score
                    "best_cnn_affinity": affinity_scores[0] if affinity_scores else None, # best affinity
                    "num_poses": max(len(pose_scores), len(affinity_scores)), # number of total poses
                }
            )

    return rows


def plot_metric_histogram(values, title, xlabel, out_file: Path):
    """
    Histogram plotting function for visuliazing RMSD, CNNpose score, and CNNaffinity. 
    """
    values = [v for v in values if v is not None]

    if not values:
        print(f"Skipping plot {title}: no values found")
        return

    plt.figure()
    plt.hist(values, bins=20) # histogram plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()



def parse_training_log_metrics(log_file: Path):
    """
    Parse all GNINA-Torch train metrics from training.log.
    INcluding: accuracy, balanced accuracy, pose loss, ROC AUC, MAE, RMSE, affinity loss, epoch time, elapsed time.
    """
    metrics = {} # dict format

    if not log_file.exists():
        print(f"skipping: training log not found: {log_file}")
        return metrics

    key_map = { # dict structure
        "Accuracy": "accuracy",
        "Balanced Accuracy": "balanced_accuracy",
        "Pose Loss": "pose_loss",
        "ROC AUC": "roc_auc",
        "MAE": "mae",
        "RMSE": "rmse",
        "Affinity Loss": "affinity_loss",
        "Loss": "loss",
        "Epoch Time": "epoch_time",
        "Elapsed Time": "elapsed_time",
    }

    current_epoch = None

# use regex expressions for flexibility
    with log_file.open(errors="ignore") as f:
        for line in f:
            line = line.strip()

            train_match = re.search( # find the training results per epoch (find headers)
                r">>>\s*Train Results\s*-\s*Epoch\[(\d+)\]",
                line,
                re.IGNORECASE,
            )

            if train_match: # if there is a header line, create a new dict entry for the metrics of each training epoch 
                current_epoch = int(train_match.group(1))
                metrics[current_epoch] = {}
                continue
            
            if current_epoch is None: # error handling
                continue

            metric_match = re.match( # for each line, find out if a metric in the dict is mentioned
                r"(.+?):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                line,
            )

            if metric_match: # get the titles (raw_key) of each line and its contents (value)
                raw_key = metric_match.group(1).strip()
                value = float(metric_match.group(2))

                if raw_key in key_map: # if the raw_key is a valid metric in the dict, put the metric and its value into the metrics dict
                    metrics[current_epoch][key_map[raw_key]] = value

    return metrics


def plot_training_loss(training_out_dir: Path, out_file: Path):

    log_file = training_out_dir / "training.log"

    metrics = parse_training_log_metrics(log_file) # dict 

    if not metrics:
        print("skipping loss plot: no training metrics found")
        return

    epochs = sorted(metrics.keys()) # sorted across epochs 

    pose_loss = [
        metrics[e]["pose_loss"]
        for e in epochs
        if "pose_loss" in metrics[e]
    ]

    affinity_loss = [
        metrics[e]["affinity_loss"]
        for e in epochs
        if "affinity_loss" in metrics[e]
    ]

    total_loss = [ #additive: pose loss + affinity loss 
        metrics[e]["loss"]
        for e in epochs
        if "loss" in metrics[e]
    ]

    plt.figure()
    # in case we just want to plot pose loss:
    if pose_loss:
        plt.plot(
            epochs,
            pose_loss,
            marker="o",
            label="Pose Loss",
        )
    # in case we just want to plot affinity loss: 
    if affinity_loss:
        plt.plot(
            epochs,
            affinity_loss,
            marker="o",
            label="Affinity Loss",
        )
    # additive loss:
    if total_loss:
        plt.plot(
            epochs,
            total_loss,
            marker="o",
            label="Total Loss",
        )

    plt.title("GNINA-Torch Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_file, dpi=300)
    plt.close()


def write_metrics_csv(rows, out_csv: Path):
    """
    Function to extract collected docking metrics to a CSV file. Uses rows generated by 'collect_metrics' and output CSV file path (out_csv).
    This is so that we can view the metrics better, as a reader, in the dir of each model. 
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "complex_id",
        "rmsd",
        "best_cnn_pose_score",
        "best_cnn_affinity",
        "num_poses",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(cli_args, trained_model_dirs):
    plot_dir = cli_args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    # loop over models that were trained and make plots for the metrics of each model 
    for model_dir in trained_model_dirs:

        model_name = model_dir.name

        # training loss plot
        plot_training_loss(
            model_dir,
            plot_dir / f"{model_name}_training_loss.png",
        )

        # directory w/ redocked SDFs
        model_redock_dir = (
            cli_args.retrained_test_out_dir / model_name
        )
        # put the metrics into a CSV to view easier
        test_rows = collect_metrics(
            cli_args.csv,
            cli_args.input_dir,
            model_redock_dir,
            model_name,
        )

        print(
            f"Collected {len(test_rows)} rows for {model_name}"
        )
        # write CSV of model metrics
        write_metrics_csv(
            test_rows,
            plot_dir / f"{model_name}_metrics.csv",
        )
        # RMSD plot
        plot_metric_histogram(
            [row["rmsd"] for row in test_rows],
            f"{model_name} RMSD Distribution",
            "RMSD Å",
            plot_dir / f"{model_name}_rmsd_distribution.png",
        )
        # CNN pose score plot
        plot_metric_histogram(
            [row["best_cnn_pose_score"] for row in test_rows],
            f"{model_name} CNN Pose Score Distribution",
            "CNN pose score",
            plot_dir / f"{model_name}_cnn_pose_distribution.png",
        )
        # CNN affinity plot
        plot_metric_histogram(
            [row["best_cnn_affinity"] for row in test_rows],
            f"{model_name} CNN Affinity Distribution",
            "CNN affinity",
            plot_dir / f"{model_name}_cnn_affinity_distribution.png",
        )

    print(
        f"Saved all plots and metrics to: {plot_dir}"
    )

def main():
    """
    Runs the main GNINA workflow by including docking on training datasets, .types file generation, GNINA-Torch training, redocking, and plots generation.
    """
    total_start = time.time()

    cli_args = get_args()

    trained_model_dirs = [] # each trained model gets a dir in the format of [model name]_seed[seed number]

    # Step 1: Docking the training set with GNINA
    run_dataset(cli_args.train_csv, cli_args.train_input_dir, cli_args.train_out_dir, cli_args.run, "TRAIN DATA")

    # Step 2: Generating .types files for train data. LATER: generate .types files for validation set
    if cli_args.make_types:
        t0 = time.time()
        build_types(cli_args.train_csv, cli_args.train_input_dir, cli_args.train_out_dir, cli_args.train_types)
        print(f"Building .types files took {time.time() - t0:.1f} seconds")
    else:
        print("\nSkipping .types preprocessing.")

    # LATER: Validation .types generation
  #  validation_types = None

 #   if cli_args.val_csv is not None and cli_args.val_input_dir is not None:
 #       val_out_dir = Path("results/proto_val")
#        run_dataset(cli_args.val_csv, cli_args.val_input_dir, val_out_dir, cli_args.run, "VALIDATION DATA")
 #       build_types(cli_args.val_csv, cli_args.val_input_dir, val_out_dir, cli_args.val_types)
 #       validation_types = cli_args.val_types
 #   else:
 #       validation_types = cli_args.val_types if cli_args.val_types.exists() else None

    # Step 3: Training GNINA-Torch model variants from the generated train .types
    if cli_args.train_model:
        t1 = time.time()
        for model_name in cli_args.models: # across model types
            for seed in cli_args.seeds: # across the different seeds used
                run_name = f"{model_name}_seed{seed}" # the output dir is named after the run_name
                model_out_dir = cli_args.training_out_dir / run_name
                model_log = model_out_dir / "training.log"

                run_training( # train the model variants 
                    cli_args.train_types,
                    None,  # later: cli_args.val_types
                    cli_args.epochs,
                    model_out_dir,
                    model_log,
                    model_name,
                    seed,
                )
                trained_model_dirs.append(model_out_dir)

        print(f"Training step took {time.time() - t1:.1f} seconds")

        # loop over trained_model_dirs to redock over all retrained models
        for model_dir in trained_model_dirs:

            model_redock_dir = (
                cli_args.retrained_test_out_dir /
                model_dir.name
            )

            redock_test_with_retrained_model( # redock over all retrained models
                cli_args.csv,
                cli_args.input_dir,
                model_redock_dir,
                cli_args.run,
                model_dir,
            )

    else:
        print("Skipping model training.")

    print(f"Total runtime: {time.time() - total_start:.1f} seconds")

    # Step 4: Generating plots from outputs
    if cli_args.make_plots:
        make_plots(
            cli_args,
            trained_model_dirs,
        )
    else:
        print("Skipping plots.")


if __name__ == "__main__":
    main()
