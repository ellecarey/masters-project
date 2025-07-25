import yaml
import argparse
import subprocess
import sys
from pathlib import Path

from src.data_generator_module.utils import find_project_root

def prompt_for_data_config(project_root: Path):
    """
    Scans for data configs, prompts the user, and returns an absolute path.
    """
    config_dir = project_root / "configs" / "data_generation"
    try:
        available_configs = sorted(list(config_dir.glob("*.yml")))
        available_configs.extend(sorted(list(config_dir.glob("*.yaml"))))
    except FileNotFoundError:
        print(f"Error: Configuration directory '{config_dir}' not found.")
        return None

    if not available_configs:
        print(f"Error: No data configuration files found in '{config_dir}'.")
        return None

    print("\nPlease select a data configuration to run the tuning job on:")
    for i, path in enumerate(available_configs):
        print(f"  [{i + 1}] {path.name}")

    while True:
        try:
            choice = input(f"\nEnter the number of the config to use (1-{len(available_configs)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_configs):
                selected_path = available_configs[choice_idx]
                print(f"You selected: {selected_path.name}")
                return selected_path
            else:
                print("Invalid number. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number from the list.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled. Exiting.")
            return None

def main():
    """
    Reads experiments.yml, prompts for a data config, and launches a 
    parallel tuning job using robust, absolute file paths.
    """
    parser = argparse.ArgumentParser(description="Experiment runner for hyperparameter tuning.")
    parser.add_argument("--job", "-j", type=str, required=True, help="The name of the tuning job to run from experiments.yml.")
    args = parser.parse_args()

    try:
        project_root = Path(find_project_root())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    experiments_file = project_root / "configs" / "experiments.yml"
    try:
        with open(experiments_file, "r") as f:
            experiments_config = yaml.safe_load(f)
            tuning_jobs = experiments_config.get("tuning_jobs", {})
    except FileNotFoundError:
        print(f"Error: `{experiments_file}` not found.")
        sys.exit(1)

    job_params = tuning_jobs.get(args.job)
    if not job_params:
        print(f"Error: Job '{args.job}' not found in `{experiments_file}`.")
        sys.exit(1)

    selected_data_config = prompt_for_data_config(project_root)
    if not selected_data_config:
        sys.exit(1)

    print(f"\n--- Starting Job: {args.job} ---")
    print(f"Description: {job_params.get('description', 'N/A')}")
    print(f"Number of parallel workers: {job_params['num_workers']}")
    print("-" * 35)

    command = [
        "uv", "run", "run_hyperparameter_tuning.py",
        "--data-config", str(selected_data_config.resolve()),
        "--tuning-config", str((project_root / job_params["tuning_config"]).resolve()),
        "--base-training-config", str((project_root / job_params["base_training_config"]).resolve()),
        "--n-trials", str(job_params["n_trials"]),
        "--sample-fraction", str(job_params["sample_fraction"]),
    ]

    workers = []
    for i in range(job_params["num_workers"]):
        print(f"Launching worker {i + 1}...")
        process = subprocess.Popen(command, cwd=project_root)
        workers.append(process)

    print(f"\nAll {len(workers)} workers launched. Waiting for tuning to complete...")
    for i, worker in enumerate(workers):
        worker.wait()
        print(f"Worker {i + 1} has finished (Exit code: {worker.returncode}).")
    
    print("\n--- All workers have finished. Job complete. ---")

    print("\n" + "="*80)
    print("Tuning phase is complete. To analyse the results and select the best trial, run:")
    analysis_command = (
        f"uv run run_tuning_analysis.py "
        f"-dc {selected_data_config.resolve()} "
        f"-btc {(project_root / job_params['base_training_config']).resolve()}"
    )
    print(analysis_command)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
