def run_hyperparameter_tuning(
    data_config: str,
    tuning_config: str,
    base_training_config: str,
    sample_fraction: float = 0.8,
    n_trials: int = 50
):
    """
    Run Optuna hyperparameter tuning on a dataset given a search space config.
    """
    import optuna
    import pandas as pd
    from pathlib import Path
    import copy
    import torch
    import os

    from src.data_generator_module import utils as data_utils
    from src.training_module.models import get_model
    from src.training_module.dataset import TabularDataset
    from src.training_module.trainer import train_model
    from sklearn.model_selection import train_test_split
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Configs and Data ---
    tuning_config_dict = data_utils.load_yaml_config(tuning_config)
    data_config_dict = data_utils.load_yaml_config(data_config)
    dataset_base_name = data_utils.create_filename_from_config(data_config_dict)
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")
    try:
        full_data = pd.read_csv(dataset_filepath)
        print(f"Loaded full data from: {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_filepath}. Please generate it first.")
        return

    # --- Connect to the Optuna Study ---
    model_name_suffix = Path(base_training_config).stem
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///db/{study_name}_tuning.db"
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name,
    )
    print(f"\nWorker starting on study '{study_name}' on device: '{device}'")

    # --- Objective Function ---
    def objective(trial):
        # Sample data
        X_full = full_data.drop(columns=["target"])
        y_full = full_data["target"]
        if sample_fraction >= 1.0:
            X_sample, y_sample = X_full, y_full
        else:
            X_sample, _, y_sample, _ = train_test_split(
                X_full, y_full, train_size=sample_fraction, stratify=y_full, random_state=trial.number
            )

        # Hyperparameter search
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

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=trial.number, stratify=y_sample
        )
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        import time
        start_time = time.time()
        try:
            trained_model, _ = train_model(
                model=model,
                train_loader=train_loader,
                validation_loader=val_loader,
                criterion=criterion,
                optimiser=optimiser,
                epochs=epochs,
                device=device,
                trial=trial,
            )
        except optuna.TrialPruned:
            raise
        training_time = time.time() - start_time
        trial.set_user_attr("training_time", training_time)

        trained_model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = trained_model(features)
                scores = torch.sigmoid(outputs)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_scores)
        return auc

    # --- Run the Optuna Optimisation ---
    study.optimize(
        objective,
        n_trials=n_trials,
    )
    print(f"\nWorker has finished its trials for study '{study_name}'.")

def run_experiments(job: str):
    """
    Launch (in parallel) all workers for a multi-worker Optuna tuning job,
    monitor progress, show step-by-step feedback, as in run_experiments.py.
    """
    import yaml
    import subprocess
    import sys
    import time
    from pathlib import Path
    import optuna
    from tqdm import tqdm

    # Find project root and experiments config
    from src.data_generator_module.utils import find_project_root, create_filename_from_config

    # --- Load experimental job config ---
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

    job_params = tuning_jobs.get(job)
    if not job_params:
        print(f"Error: Job '{job}' not found in `{experiments_file}`.")
        sys.exit(1)

    # --- Prompt user to select a data config ---
    config_dir = project_root / "configs" / "data_generation"
    available_configs = sorted(list(config_dir.glob("*.yml"))) + sorted(list(config_dir.glob("*.yaml")))
    if not available_configs:
        print(f"Error: No data configuration files found in '{config_dir}'.")
        sys.exit(1)

    print("\nPlease select a data configuration to run the tuning job on:")
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

    # --- Prepare Distributed Study ---
    print("\n--- Preparing Distributed Study ---")
    with open(selected_data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    dataset_base_name = create_filename_from_config(data_config)
    base_training_config_path = project_root / job_params["base_training_config"]
    model_name_suffix = base_training_config_path.stem

    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///db/{study_name}_tuning.db"

    optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    print(f"Successfully created or loaded study '{study_name}' in '{storage_name}'")

    # --- Launch Parallel Workers ---
    print(f"\n--- Starting Job: {job} ---")
    print(f"Number of parallel workers: {job_params['num_workers']}")
    print("-" * 35)
    n_trials_per_worker = (job_params['n_trials'] + job_params['num_workers'] - 1) // job_params['num_workers']
    command = [
        "uv",
        "run",
        "experiment_manager.py",
        "tune-worker",
        "--data-config", str(selected_data_config_path),
        "--tuning-config", str((project_root / job_params["tuning_config"]).resolve()),
        "--base-training-config", str(base_training_config_path.resolve()),
        "--n-trials", str(n_trials_per_worker),
        "--sample-fraction", str(job_params["sample_fraction"]),
    ]
    workers = []
    for i in range(job_params['num_workers']):
    # uncomment this section for debugging to show the output of one worker
    #      if i == 0:
    #         # Let the first worker print its output to the console
    #         print("Launching worker 1 with output enabled for debugging...")
    #         process = subprocess.Popen(command, cwd=project_root) # No output redirection
    # else: 
    # end of debugging
        process = subprocess.Popen(command, cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        workers.append(process)
    print(f"All {len(workers)} workers launched. Monitoring study progress...")

    # --- Monitor progress with tqdm ---
    total_trials_to_run = job_params['n_trials']
    from optuna.trial import TrialState
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
                all_workers_exited = all(p.poll() is not None for p in workers)
                if len(finished_trials) >= total_trials_to_run or (all_workers_exited and len(running_trials) == 0):
                    pbar.n = total_trials_to_run
                    pbar.refresh()
                    break
                time.sleep(2)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Stopping monitoring.")
                break
            except ValueError as e:
                if "Record does not exist" in str(e):
                    time.sleep(1)
                    continue
                else:
                    print(f"\nAn unexpected ValueError occurred: {e}")
                    break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                break
    pbar.set_postfix_str("Complete")
    print("\n--- All workers have finished. Job complete. ---")
    print("\n" + "="*80)
    print("Tuning phase is complete. To analyse the results and select the best trial, run:")
    analysis_command = (
        f"uv run experiment_manager.py tune-analysis "
        f"--data-config {selected_data_config_path} "
        f"--base-training-config {base_training_config_path.resolve()}"
    )
    print(analysis_command)
    print("="*80 + "\n")

def run_tuning_analysis(data_config: str, base_training_config: str):
    """
    Analyse a completed Optuna study, interactively select the optimal trial, and
    generate an optimal training config. Mirrors run_tuning_analysis.py.
    """
    import os
    import yaml
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    import optuna
    from optuna.trial import TrialState
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    import warnings

    from src.data_generator_module import utils as data_utils

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    # --- Find root, load configs ---
    project_root = Path(data_utils.find_project_root())
    data_config_path = Path(data_config)
    base_training_config_path = Path(base_training_config)
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
    print(f"--- Analysing Study: {study_name} ---")
    print(f"Loading results from: {storage_name}")

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"Error: Study '{study_name}' not found in the database.")
        return
    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(completed_trials) == 0:
        print("Error: No trials in the study completed successfully.")
        return

    # --- Show top 5 trials ---
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed_trials[:5]
    print("\n--- Top 5 Completed Trials (by AUC) ---")
    for i, trial in enumerate(top_trials):
        training_time = trial.user_attrs.get('training_time', 'N/A')
        time_str = f"{training_time:.2f}" if isinstance(training_time, float) else "N/A"
        print(f"{i+1: <2} | Trial #{trial.number: <4} | AUC: {trial.value:.4f} | Time: {time_str} s | {json.dumps(trial.params)}")

    # --- User selects optimal trial ---
    selected_trial = None
    while not selected_trial:
        try:
            choice_str = input(f"\nEnter the Rank of the trial to use (1-{len(top_trials)}): ")
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(top_trials):
                selected_trial = top_trials[choice_idx]
                print(f"You selected Rank {choice_str} (Trial #{selected_trial.number}).")
            else:
                print(f"Invalid rank. Please enter a number between 1 and {len(top_trials)}.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number from the list.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled. Exiting.")
            return

    # --- Visualisations ---
    model_name = tuning_config.get("model_name", "model")

    # Extract family base from dataset_base_name (remove _seed\d+)
    from src.utils.report_paths import extract_family_base
    family_base = extract_family_base(dataset_base_name)
    
    # Create tuning-specific directory at family level
    output_plot_dir = project_root / "reports" / "figures" / family_base / f"{model_name}_tuning"
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"\nSaving plots to: {output_plot_dir}")
    plt.figure()
    plot_optimization_history(study, target_name="AUC")
    plt.savefig(output_plot_dir / "tuning_optimisation_history_auc.pdf", bbox_inches='tight')
    plt.close()

    try:
        plt.figure()
        plot_param_importances(study, target_name="AUC")
        plt.savefig(output_plot_dir / "tuning_param_importances_auc.pdf", bbox_inches='tight')
        plt.close()
    except Exception:
        print("Warning: Could not generate parameter importance plot.")

    # --- Write Final Training Config ---
    def create_and_save_optimal_config(best_params, final_data_config_path, base_training_config_path, tuning_config):
        with open(base_training_config_path, 'r') as f:
            config_template = yaml.safe_load(f)
        if 'model_name' in tuning_config:
            config_template['training_settings']['model_name'] = tuning_config['model_name']
        config_template['training_settings']['hyperparameters'].update(best_params)
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
    )

    # --- Print next steps ---
    print("\n" + "="*80)
    print("Next Step: Choose a final training method:")
    print("="*80)
    print("\nOPTION 1: Train a single model on this specific dataset")
    print(f"uv run experiment_manager.py train-multiseed --data-config-base {data_config_path} --optimal-config {optimal_config_path}")
    print("\nOPTION 2: Train models on ALL datasets in this family (all seeds)")
    print(f"uv run experiment_manager.py train-multiseed --data-config-base {data_config_path} --optimal-config {optimal_config_path}")
    print("\n" + "="*80)
