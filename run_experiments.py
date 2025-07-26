import yaml
import argparse
import subprocess
import sys
import time
from pathlib import Path
import optuna
from tqdm import tqdm
from optuna.trial import TrialState
from src.data_generator_module.utils import find_project_root, create_filename_from_config

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
        print(f" [{i + 1}] {path.name}")

    while True:
        try:
            choice = input(f"\nEnter the number of the config to use (1-{len(available_configs)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_configs):
                selected_path = available_configs[choice_idx]
                print(f"You selected: {selected_path.name}")
                return selected_path.resolve()
            else:
                print("Invalid number. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number from the list.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled. Exiting.")
            return None

def main():
    """
    Reads experiments.yml, creates the Optuna study, launches parallel
    tuning jobs, and monitors the overall progress with a progress bar.
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

    selected_data_config_path = prompt_for_data_config(project_root)
    if not selected_data_config_path:
        sys.exit(1)

    # --- Prepare Distributed Study ---
    print("\n--- Preparing Distributed Study ---")
    with open(selected_data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    
    dataset_base_name = create_filename_from_config(data_config)
    base_training_config_path = project_root / job_params["base_training_config"]
    
    # Extract model name suffix from the config filename
    model_name_suffix = base_training_config_path.stem
    
    # Construct study name and storage path, now including the model name
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///reports/{dataset_base_name}_{model_name_suffix}_tuning.db"
    
    optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    print(f"Successfully created or loaded study '{study_name}' in '{storage_name}'")

   # --- Launch Parallel Workers ---
    print(f"\n--- Starting Job: {args.job} ---")
    print(f"Number of parallel workers: {job_params['num_workers']}")
    print("-" * 35)

    n_trials_per_worker = (job_params['n_trials'] + job_params['num_workers'] - 1) // job_params['num_workers']
    command = [
        "uv", "run", "run_hyperparameter_tuning.py",
        "--data-config", str(selected_data_config_path),
        "--tuning-config", str((project_root / job_params["tuning_config"]).resolve()),
        "--base-training-config", str(base_training_config_path.resolve()),
        "--n-trials", str(n_trials_per_worker),
        "--sample-fraction", str(job_params["sample_fraction"]),
    ]
    workers = []
    for i in range(job_params['num_workers']):
        process = subprocess.Popen(command, cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        workers.append(process)
    print(f"All {len(workers)} workers launched. Monitoring study progress...")

    # --- Monitor Progress by Polling the Database ---
    total_trials_to_run = job_params['n_trials']
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    
    with tqdm(total=total_trials_to_run, desc="Overall Tuning Progress", bar_format=bar_format) as pbar:
        finished_trial_count = 0
        while finished_trial_count < total_trials_to_run:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                
                all_trials = study.get_trials(deepcopy=False)
                finished_trials = [t for t in all_trials if t.state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]]
                
                pbar.update(len(finished_trials) - finished_trial_count)
                finished_trial_count = len(finished_trials)

                best_value = None
                if study.best_trial and study.best_trial.state == TrialState.COMPLETE:
                    best_value = study.best_value
                
                running_trials = [t for t in all_trials if t.state == TrialState.RUNNING]
                postfix_str = (
                    f"Running: {len(running_trials)}, "
                    f"Pruned: {len([t for t in finished_trials if t.state == TrialState.PRUNED])}, "
                    f"Best AUC: {best_value:.4f}" if best_value is not None else "Best AUC: N/A"
                )
                pbar.set_postfix_str(postfix_str)

                if all(p.poll() is not None for p in workers) and not running_trials:
                    if finished_trial_count >= total_trials_to_run:
                        break
                
                time.sleep(2)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Stopping monitoring.")
                break
            except ValueError as e:
                # THIS IS THE KEY CHANGE: Silently ignore the expected race condition error
                if "Record does not exist" in str(e):
                    time.sleep(1) # Wait a moment and let the loop retry
                    continue
                else:
                    print(f"\nAn unexpected ValueError occurred: {e}")
                    break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                break
        
        if pbar.n < total_trials_to_run:
            pbar.update(total_trials_to_run - pbar.n)
        pbar.set_postfix_str("Complete")

    print("\n--- All workers have finished. Job complete. ---")
    
    print("\n" + "="*80)
    print("Tuning phase is complete. To analyse the results and select the best trial, run:")
    analysis_command = (
        f"uv run run_tuning_analysis.py "
        f"-dc {selected_data_config_path} "
        f"-btc {base_training_config_path.resolve()}"
    )
    print(analysis_command)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()