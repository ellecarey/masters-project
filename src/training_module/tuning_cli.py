import yaml
import subprocess
import sys
import time
from pathlib import Path
import optuna
from tqdm import tqdm
import pandas as pd
import copy
import torch
import os
import json
import shutil
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data_generator_module import utils as data_utils
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_generator_module.utils import find_project_root, create_filename_from_config
from src.utils.report_paths import extract_family_base
from src.utils.plotting_helpers import generate_subtitle_from_config
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.training_module.utils import plot_training_history, plot_final_metrics, plot_combined_training_histories


def train_candidate_worker(args):
    """
    Worker function for training a single candidate on a sampled dataset.
    """
    trial_data, i, tuning_config_data, dataset_filepath, output_plot_dir, \
    model_name, scheduler_settings, early_stopping_settings, sample_fraction = args

    # --- Data loading and splitting inside the worker ---
    full_data = pd.read_csv(dataset_filepath)
    target_column = "target"
    X = full_data.drop(columns=[target_column])
    y = full_data[target_column]

    trial_number = trial_data['number']

    if sample_fraction < 1.0:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_fraction, random_state=trial_number, stratify=y
        )
    else:
        X_sample, y_sample = X, y

    # Further split the sample into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    # --- End of data loading block ---

    trial_params = trial_data['params']
    training_time = trial_data.get('training_time', 'N/A')

    arch_model_name = tuning_config_data.get("model_name", "mlp_001")
    ARCH_PARAMS = {"mlp_001": {"hidden_size"}}
    model_params = {
        key: trial_params[key]
        for key in trial_params
        if key in ARCH_PARAMS.get(arch_model_name, set())
    }
    model_params["input_size"] = X_train.shape[1]
    model_params["output_size"] = 1

    model = get_model(arch_model_name, model_params)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=trial_params["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=trial_params["batch_size"], shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["learning_rate"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_settings.get('mode', 'min'),
        factor=scheduler_settings.get('factor', 0.1),
        patience=scheduler_settings.get('patience', 5)
    )
    patience = early_stopping_settings.get("patience", 20)

    trained_model, history, _ = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        criterion=criterion,
        optimiser=optimizer,
        epochs=trial_params["epochs"],
        device=device,
        verbose=False,
        scheduler=scheduler,
        early_stopping_enabled=True,
        patience=patience
    )

    plot_subtitle = f"Candidate {i+1} - Trial #{trial_number}\nParams: {json.dumps(trial_params)}"
    history_plot_name = f"{model_name}_training_history_trial{trial_number}"

    plot_training_history(
        history=history,
        experiment_name=history_plot_name,
        output_dir=output_plot_dir,
        subtitle=plot_subtitle
    )

    plot_final_metrics(
        model=trained_model,
        test_loader=val_loader,
        device=device,
        model_name=model_name,
        trial_number=trial_number,
        output_dir=output_plot_dir,
        subtitle=plot_subtitle
    )

    candidate_key = f"candidate_{i+1}_trial_{trial_number}"
    time_str = f"{training_time:.2f}" if isinstance(training_time, float) else "N/A"

    return {
        'rank': i+1,
        'trial_number': trial_data['number'],
        'trial_value': trial_data['value'],
        'trial_params': trial_params,
        'auc': trial_data['value'],
        'params': trial_params,
        'training_time': time_str,
        'model_key': candidate_key,
        'model_state': trained_model.state_dict(),
        'history': history,
        'X_train_shape': X_train.shape
    }

def run_hyperparameter_tuning(
    data_config: str,
    tuning_config: str,
    base_training_config: str,
    sample_fraction: float = 0.8,
    n_trials: int = 50
):
    apply_custom_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tuning_config_dict = data_utils.load_yaml_config(tuning_config)
    data_config_dict = data_utils.load_yaml_config(data_config)
    dataset_base_name = data_utils.create_filename_from_config(data_config_dict)
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")

    try:
        full_data = pd.read_csv(dataset_filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_filepath}.")
        return

    model_name_suffix = Path(base_training_config).stem
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///db/{study_name}_tuning.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(f"\nWorker starting on study '{study_name}' on device: '{device}'")

    def objective(trial):
        X_full = full_data.drop(columns=["target"])
        y_full = full_data["target"]

        if sample_fraction < 1.0:
            X_sample, _, y_sample, _ = train_test_split(
                X_full, y_full, train_size=sample_fraction, stratify=y_full, random_state=trial.number
            )
        else:
            X_sample, y_sample = X_full, y_full

        search_space = tuning_config_dict["search_space"]
        hyperparams = {}
        for param_name, params_config in search_space.items():
            params = copy.copy(params_config)
            param_type = params.pop("type")
            if param_type == "int":
                hyperparams[param_name] = trial.suggest_int(param_name, **params)
            elif param_type == "float":
                hyperparams[param_name] = trial.suggest_float(param_name, **params)
            elif param_type == "categorical":
                hyperparams[param_name] = trial.suggest_categorical(param_name, **params)

        epochs = hyperparams["epochs"]
        model_name = tuning_config_dict["model_name"]
        ARCH_PARAMS = {"mlp_001": {"hidden_size"}}
        model_params = {key: hyperparams[key] for key in hyperparams if key in ARCH_PARAMS.get(model_name, set())}
        model_params["input_size"] = X_sample.shape[1]
        model_params["output_size"] = 1

        model = get_model(model_name, model_params)

        data_utils.set_global_seed(trial.number)

        X_train, X_val, y_train, y_val = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=trial.number, stratify=y_sample
        )

        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"] 
        )
        start_time = time.time()
        try:
            trained_model, history, best_epoch = train_model(
                model=model,
                train_loader=train_loader,
                validation_loader=val_loader,
                criterion=criterion,
                optimiser=optimiser,
                epochs=epochs,
                device=device,
                trial=trial,
                early_stopping_enabled=True,
                patience=tuning_config_dict.get("early_stopping_settings", {}).get("patience", 20)
            )
        except optuna.TrialPruned:
            raise
        training_time = time.time() - start_time
        trial.set_user_attr("training_time", training_time)
        trial.set_user_attr("best_epoch", best_epoch)

        trained_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = trained_model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Return the validation loss for minimization.
        return avg_val_loss

    study.optimize(objective, n_trials=n_trials)
    print(f"\nWorker has finished its trials for study '{study_name}'.")


def run_experiments(job: str, data_config_path: str = None):
    """
    Launches a distributed hyperparameter tuning job for a given experiment setup.
    Can either use a provided data config file or interactively prompt the user.
    """
    try:
        project_root = Path(find_project_root())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Determine the data configuration file to use ---
    selected_data_config_path = None
    if data_config_path:
        # Use the path provided as an argument
        provided_path = Path(data_config_path)
        # Ensure the path is absolute or resolve it relative to the project root
        if provided_path.is_absolute():
            selected_data_config_path = provided_path
        else:
            selected_data_config_path = project_root / data_config_path

        if not selected_data_config_path.exists():
            print(f"Error: Provided data config file not found at '{selected_data_config_path}'")
            sys.exit(1)
        print(f"Using provided data config: {selected_data_config_path.name}")
    else:
        # Fallback to interactive selection
        config_dir = project_root / "configs" / "data_generation"
        available_configs = [
            path for path in sorted(list(config_dir.glob("*.yml"))) if "_training_config.yml" in path.name
        ]

        if not available_configs:
            print(f"Error: No '_training' data configuration files found in '{config_dir}'.")
            sys.exit(1)

        if len(available_configs) == 1:
            selected_data_config_path = available_configs[0]
            print(f"Automatically selected the only available training config: {selected_data_config_path.name}")
        else:
            print("\nPlease select a _training data configuration to run the tuning job on:")
            for i, path in enumerate(available_configs):
                print(f" [{i + 1}] {path.name}")
            while True:
                try:
                    choice = input(f"\nEnter the number of the config to use (1-{len(available_configs)}): ")
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_configs):
                        selected_data_config_path = available_configs[choice_idx]
                        print(f"You selected: {selected_data_config_path.name}")
                        break
                    print("Invalid number. Please try again.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter a number from the list.")
                except (KeyboardInterrupt, EOFError):
                    print("\nSelection cancelled. Exiting.")
                    sys.exit(1)

    # --- Load experiment and job configurations ---
    experiments_file = project_root / "configs" / "experiments.yml"
    try:
        with open(experiments_file, "r") as f:
            experiments_config = yaml.safe_load(f)
        tuning_jobs = experiments_config.get("tuning_jobs", {})
    except FileNotFoundError:
        print(f"Error: `{experiments_file}` not found.")
        sys.exit(1)

    job_params = tuning_jobs.get(job)
    if not job_params:
        print(f"Error: Job '{job}' not found in `{experiments_file}`.")
        sys.exit(1)

    # --- Prepare Distributed Study ---
    print("\n--- Preparing Distributed Study ---")
    with open(selected_data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    dataset_base_name = create_filename_from_config(data_config)
    base_training_config_path = project_root / job_params["base_training_config"]
    model_name_suffix = base_training_config_path.stem
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///db/{study_name}_tuning.db"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
    )

    print(f"Successfully created or loaded study '{study_name}' in '{storage_name}' for MINIMIZATION")

    # --- Launch Worker Processes ---
    print(f"\n--- Starting Job: {job} ---")
    n_trials_per_worker = (job_params['n_trials'] + job_params['num_workers'] - 1) // job_params['num_workers']
    command = [
        "uv", "run", "experiment_manager.py", "tune-worker",
        "--data-config", str(selected_data_config_path),
        "--tuning-config", str((project_root / job_params["tuning_config"]).resolve()),
        "--base-training-config", str(base_training_config_path.resolve()),
        "--n-trials", str(n_trials_per_worker),
        "--sample-fraction", str(job_params["sample_fraction"]),
    ]

    # uncomment for worker debugging
    # workers = [
    #     subprocess.Popen(command, cwd=project_root)
    #     for _ in range(job_params['num_workers'])
    # ]

    # print(f"All {len(workers)} workers launched. Monitoring study progress...")
    
    workers = [
        subprocess.Popen(command, cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(job_params['num_workers'])
    ]

    print(f"All {len(workers)} workers launched. Monitoring study progress...")
    with tqdm(total=job_params['n_trials'], desc="Overall Tuning Progress") as pbar:
        finished_trial_count = 0
        while finished_trial_count < job_params['n_trials']:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                finished_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL])
                pbar.update(len(finished_trials) - finished_trial_count)
                finished_trial_count = len(finished_trials)
                time.sleep(2)
            except Exception:
                break

    print("\n--- All workers have finished. Job complete. ---")

    # --- Display Next Steps ---
    print("\n" + "="*80)
    print("Tuning phase is complete. To analyse the results and select the best trial, run:")
    analysis_command = (
        f"uv run experiment_manager.py tune-analysis \\\n"
        f" --data-config {selected_data_config_path} \\\n"
        f" --base-training-config {base_training_config_path.resolve()} \\\n"
        f" --sample-fraction {job_params['sample_fraction']}"
    )
    print(analysis_command)
    print("="*80 + "\n")



def run_tuning_analysis(data_config: str, base_training_config: str, sample_fraction: float, non_interactive: bool = False):
    """
    Analyse a completed Optuna study, generate plots, and allow interactive selection.
    Retrains the final selected model on the full dataset.
    """
    apply_custom_plot_style()
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    project_root = Path(data_utils.find_project_root())
    data_config_path = Path(data_config)
    base_training_config_path = Path(base_training_config)

    base_train_config_dict = data_utils.load_yaml_config(base_training_config_path)
    scheduler_settings = base_train_config_dict.get("training_settings", {}).get("scheduler_settings", {})
    early_stopping_settings = base_train_config_dict.get("training_settings", {}).get("early_stopping_settings", {})

    data_config_dict = data_utils.load_yaml_config(data_config_path)
    tuning_config_path = project_root / "configs" / "tuning" / f"{base_training_config_path.stem}.yml"
    try:
        tuning_config = data_utils.load_yaml_config(tuning_config_path)
    except FileNotFoundError:
        print(f"Error: Tuning configuration not found at '{tuning_config_path}'")
        return

    dataset_base_name = data_utils.create_filename_from_config(data_config_dict)
    model_name_suffix = base_training_config_path.stem
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///db/{study_name}_tuning.db"

    print("\n--- Clearing old analysis plots ---")
    base_family = extract_family_base(dataset_base_name)
    
    # Define plot directories
    output_plot_dir = project_root / "reports" / "figures" / base_family / dataset_base_name
    tuning_output_plot_dir = project_root / "reports" / "figures" / base_family / f"{model_name_suffix}_tuning"

    # Delete directories if they exist
    if output_plot_dir.exists():
        shutil.rmtree(output_plot_dir)
        print(f"Removed old directory: {output_plot_dir}")
    
    if tuning_output_plot_dir.exists():
        shutil.rmtree(tuning_output_plot_dir)
        print(f"Removed old directory: {tuning_output_plot_dir}")
        
    # Recreate the main directory for candidate plots
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {output_plot_dir}")
    
    print(f"--- Analysing Study: {study_name} ---")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"Error: Study '{study_name}' not found.")
        return

    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if not completed_trials:
        print("Error: No trials completed successfully.")
        return
        
    completed_trials.sort(key=lambda t: t.value, reverse=False)
    top_trials = completed_trials[:5]

    print("\n--- Retraining Top 5 Candidates on Sampled Data ---")
    dataset_filepath = project_root / "data" / f"{dataset_base_name}_dataset.csv"
    if not dataset_filepath.exists():
        print(f"Error: Training dataset not found at {dataset_filepath}")
        return

    base_family = extract_family_base(dataset_base_name)
    output_plot_dir = project_root / "reports" / "figures" / base_family / dataset_base_name
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {output_plot_dir}")

    trial_args = []
    for i, trial in enumerate(top_trials):
        trial_data = {
            'params': trial.params, 'number': trial.number, 'value': trial.value,
            'training_time': trial.user_attrs.get('training_time', 'N/A')
        }
        args = (
            trial_data, i, tuning_config, dataset_filepath, output_plot_dir,
            model_name_suffix, scheduler_settings, early_stopping_settings, sample_fraction
        )
        trial_args.append(args)

    candidate_info = []
    trained_models = {}
    max_workers = min(3, os.cpu_count() or 1, len(top_trials))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_rank = {executor.submit(train_candidate_worker, args): i for i, args in enumerate(trial_args)}
        
        progress_iterator = tqdm(as_completed(future_to_rank), total=len(top_trials), desc=f"Retraining Top Candidates ({sample_fraction*100:.0f}% data)")
        
        for future in progress_iterator:
            try:
                result = future.result()
                candidate_info.append(result)
                trained_models[result['model_key']] = result['model_state']
            except Exception as e:
                rank = future_to_rank[future] + 1
                progress_iterator.write(f"✗ Failed to train candidate {rank}: {e}")

    candidate_info.sort(key=lambda x: x['rank'])
    
    if candidate_info:
        plot_combined_training_histories(
            candidate_info=candidate_info,
            output_dir=str(output_plot_dir),
            model_name=model_name_suffix,
            subtitle=generate_subtitle_from_config(data_config_dict)
        )
        print(f"Saved combined training history plot to: {output_plot_dir}")


    # --- User selection ---
    if not candidate_info:
        print("\nError: No candidates were successfully trained. Cannot proceed.")
        return

    selected_trial = None
    if non_interactive:
        print("\n--- Non-interactive mode: Automatically selecting Rank 1 candidate ---")
        if candidate_info:
            selected_candidate_info = candidate_info[0]
            trial_number_to_find = selected_candidate_info['trial_number']
            selected_trial = next((t for t in top_trials if t.number == trial_number_to_find), None)
            if selected_trial:
                print(f"Automatically selected Rank 1 (Trial #{selected_trial.number}).")
            else:
                print("Error: Could not find trial details for Rank 1 candidate.")
                return
        else:
            print("Error: No candidate info available for automatic selection.")
            return
    else:
        # interactive loop
        while not selected_trial:
            try:
                choice_str = input(f"\nEnter the Rank of the trial to use (1-{len(candidate_info)}): ")
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(candidate_info):
                    selected_candidate_info = candidate_info[choice_idx]
                    trial_number_to_find = selected_candidate_info['trial_number']
                    selected_trial = next((t for t in top_trials if t.number == trial_number_to_find), None)
                    if selected_trial:
                        print(f"You selected Rank {choice_str} (Trial #{selected_trial.number}).")
                        break
                    else:
                        print(f"Error: Could not find trial for rank {choice_str}.")
                else:
                    print(f"Invalid rank. Please enter a number between 1 and {len(candidate_info)}.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a number from the list.")
            except (KeyboardInterrupt, EOFError):
                print("\nSelection cancelled. Exiting.")
                return

    if not selected_trial:
        print("Could not select a trial. Exiting.")
        return

    # --- Retrain the chosen model on the ENTIRE dataset ---
    print("\n--- Retraining selected model on the full training dataset... ---")
    full_data = pd.read_csv(dataset_filepath)
    target_column = "target"
    X_full = full_data.drop(columns=[target_column])
    y_full = full_data[target_column]
    
    print(f"Full dataset for final training contains: {len(X_full)} samples.")
    
    final_params = selected_trial.params
    best_epoch_for_final_run = selected_trial.user_attrs.get("best_epoch")
    if not best_epoch_for_final_run:
        print(f"Warning: 'best_epoch' not found in Trial #{selected_trial.number}. Falling back to the full number of epochs from the trial's parameters.")
        epochs_for_final_run = final_params["epochs"]
    else:
        epochs_for_final_run = best_epoch_for_final_run
        print(f"Using optimal epoch number from tuning: {epochs_for_final_run} epochs.")

    
    model_name_arch = tuning_config.get("model_name", "mlp_001")
    ARCH_PARAMS = {"mlp_001": {"hidden_size"}}
    model_params = {
        key: final_params[key] for key in final_params
        if key in ARCH_PARAMS.get(model_name_arch, set())
    }
    model_params["input_size"] = X_full.shape[1]
    model_params["output_size"] = 1
    final_model = get_model(model_name_arch, model_params)

    full_train_dataset = TabularDataset(X_full, y_full)
    full_train_loader = DataLoader(dataset=full_train_dataset, batch_size=final_params["batch_size"], shuffle=True)
    
    final_optimiser = torch.optim.Adam(final_model.parameters(), lr=final_params["learning_rate"])

    final_model, final_history, _ = train_model(
        model=final_model,
        train_loader=full_train_loader,
        validation_loader=None,
        criterion=nn.BCEWithLogitsLoss(),
        optimiser=final_optimiser,
        epochs=epochs_for_final_run,
        device=device,
        verbose=True,
        scheduler=None,
        early_stopping_enabled=False
    )
    print("✓ Final model training complete!")
    
    # --- Visualisations ---
    plot_subtitle = generate_subtitle_from_config(data_config_dict)
    model_name_vis = tuning_config.get("model_name", "model")
    family_base_vis = extract_family_base(dataset_base_name)
    tuning_output_plot_dir = project_root / "reports" / "figures" / family_base_vis / f"{model_name_vis}_tuning"
    os.makedirs(tuning_output_plot_dir, exist_ok=True)

    print(f"\nSaving tuning analysis plots to: {tuning_output_plot_dir}")
    # Plot Optimisation History
    plt.figure()
    # MODIFIED: Set the target name to Loss for the plot title.
    ax = plot_optimization_history(study, target_name="Validation Loss")
    ax.set_title("")
    ax.set_ylabel("LogLoss")
    main_title = "Optimisation History (Validation Loss)"
    full_title = f"{main_title}\n{plot_subtitle}"
    plt.suptitle(full_title, y=0.95)
    plt.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(tuning_output_plot_dir / "tuning_optimisation_history_loss.pdf", bbox_inches='tight')
    plt.close()

    # Plot Parameter Importances
    try:
        plt.figure()
        ax_params = plot_param_importances(study, target_name="Validation Loss")
        ax_params.set_title("")
        ax_params.set_ylabel("")
        main_title = "Hyperparameter Importances"
        full_title = f"{main_title}\n{plot_subtitle}"
        plt.suptitle(full_title, y=0.95)
        plt.tight_layout(rect=[0, 0.0, 1, 0.92])
        plt.savefig(tuning_output_plot_dir / "tuning_param_importances_loss.pdf", bbox_inches='tight')
        plt.close()
    except Exception:
        print("Warning: Could not generate parameter importance plot.")
        
    # --- Write Final Training Config ---
    def create_and_save_optimal_config(best_params, final_data_config_path, base_training_config_path, tuning_config, best_trial_number):
        with open(base_training_config_path, 'r') as f:
            config_template = yaml.safe_load(f)

        if 'model_name' in tuning_config:
            config_template['training_settings']['model_name'] = tuning_config['model_name']

        config_template['training_settings']['hyperparameters'].update(best_params)
        config_template['training_settings']['optimal_trial_number'] = best_trial_number
        
        final_data_config = data_utils.load_yaml_config(final_data_config_path)
        dataset_base_name = data_utils.create_filename_from_config(final_data_config)
        training_suffix = Path(base_training_config_path).stem
        
        new_config_filename = f"{dataset_base_name}_{training_suffix}_optimal.yml"
        save_dir = project_root / "configs" / "training" / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / new_config_filename

        with open(save_path, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, sort_keys=False)
        print(f"\nOptimal training configuration saved to: {save_path}")
        return save_path

    optimal_config_path = create_and_save_optimal_config(
        best_params=selected_trial.params,
        final_data_config_path=str(data_config_path),
        base_training_config_path=str(base_training_config_path),
        tuning_config=tuning_config,
        best_trial_number=selected_trial.number
    )

    # --- Save the final model ---
    model_output_dir = project_root / "models"
    final_model_path = model_output_dir / f"{dataset_base_name}_{model_name_vis}_optimal_model.pt"
    torch.save(final_model.state_dict(), final_model_path)
    print(f"Final model (trained on full data) saved to: {final_model_path}")

    # --- Print next steps ---
    family_base = extract_family_base(dataset_base_name)
    eval_data_config_path = project_root / "configs" / "data_generation" / f"{family_base}_seed0_config.yml"
    print("\n" + "="*80)
    print("Next Steps: Train your final model and evaluate it on all seeds")
    print("="*80)
    
    print("\nNOTE: The final model has already been trained on the full dataset and saved.")
    print("You can now proceed directly to evaluation.")

    print("\nSTEP 1: Evaluate the single trained model against all evaluation datasets (seeds 0-9).")
    print("---------------------------------------------------------------------------------------")
    eval_command = (
        f"uv run experiment_manager.py evaluate-multiseed \\\n"
        f" --trained-model {final_model_path} \\\n"
        f" --data-config-base {eval_data_config_path} \\\n"
        f" --optimal-config {optimal_config_path}"
    )
    print(eval_command)
    print("\n" + "="*80)