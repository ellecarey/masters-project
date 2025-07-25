# run_hyperparameter_tuning.py

import os
import argparse
from pathlib import Path
import pandas as pd
import time
import torch
import torch.nn as nn
import yaml
import optuna
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Keep Optuna's specific visualisation tools
from optuna.visualization import plot_pareto_front, plot_optimization_history, plot_param_importances

from src.data_generator_module import utils as data_utils
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model


def objective(trial, full_data, tuning_config, sample_size, device):
    """
    The objective function for Optuna to optimise.
    It trains a model on a sample of the data and returns performance (AUC) and training time.
    """
    # Create a stratified sample from the full dataset for this trial
    X_full = full_data.drop(columns=["target"])
    y_full = full_data["target"]
    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=sample_size, stratify=y_full, random_state=trial.number
    )

    # Suggest Hyperparameters from the search space defined in the config
    search_space = tuning_config["search_space"]
    hyperparams = {}
    for param_name, params in search_space.items():
        param_type = params["type"]
        suggestion_params = {k: v for k, v in params.items() if k != "type"}
        if param_type == "int":
            hyperparams[param_name] = trial.suggest_int(param_name, **suggestion_params)
        elif param_type == "float":
            hyperparams[param_name] = trial.suggest_float(param_name, **suggestion_params)
        elif param_type == "categorical":
            hyperparams[param_name] = trial.suggest_categorical(param_name, **suggestion_params)

    # Standard training setup on the sample
    data_utils.set_global_seed(trial.number)  # Seed for weight initialization
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=trial.number, stratify=y_sample
    )
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    # Model instantiation and training
    model_name = tuning_config["model_name"]
    # Ensure only architectural params go to the model constructor
    ARCH_PARAMS = {"MLP": {"hidden_size"}, "LogisticRegression": {}}
    valid_arch_keys = ARCH_PARAMS.get(model_name, set())
    model_params = {key: hyperparams[key] for key in hyperparams if key in valid_arch_keys}
    model_params["input_size"] = X_sample.shape[1]
    model_params["output_size"] = 1 # Assuming binary classification

    model = get_model(model_name, model_params)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Measure training time and train the model
    start_time = time.time()
    trained_model, _ = train_model(
        model, train_loader, val_loader, criterion, optimiser, hyperparams["epochs"], device
    )
    training_time = time.time() - start_time

    # Evaluate on the validation set to get the AUC score
    trained_model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = trained_model(features)
            scores = torch.sigmoid(outputs)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_scores)
    
    # Return the two objectives for Pareto optimisation
    return auc, training_time


def create_and_save_optimal_config(best_params, final_data_config_path, base_training_config_path, tuning_config):
    """Creates a new training config file populated with the best hyperparameters."""
    with open(base_training_config_path, 'r') as f:
        config_template = yaml.safe_load(f)

    # Add model name from tuning config
    if 'model_name' in tuning_config:
        config_template['training_settings']['model_name'] = tuning_config['model_name']

    # Update the hyperparameters with the best ones from the selected trial
    config_template['training_settings']['hyperparameters'].update(best_params)
    
    # Generate a descriptive name for the new config file
    final_data_config = data_utils.load_yaml_config(final_data_config_path)
    dataset_base_name = data_utils.create_filename_from_config(final_data_config)
    training_suffix = Path(base_training_config_path).stem
    new_config_filename = f"{dataset_base_name}_{training_suffix}_optimal.yml"

    save_dir = "configs/training/generated"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, new_config_filename)

    with open(save_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nOptimal training configuration saved to: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning.")
    parser.add_argument("--data-config", "-dc", required=True, help="Path to the LARGE data config file to sample from.")
    parser.add_argument("--tuning-config", "-tc", required=True, help="Path to the tuning config file with the search space.")
    parser.add_argument("--base-training-config", "-btc", required=True, help="Path to the base training config template.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of samples for each tuning trial.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of tuning trials to run.")
    args = parser.parse_args()

    # --- 1. Load Full Dataset and Configs ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuning_config = data_utils.load_yaml_config(args.tuning_config)
    data_config = data_utils.load_yaml_config(args.data_config)
    dataset_base_name = data_utils.create_filename_from_config(data_config)
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")

    try:
        full_data = pd.read_csv(dataset_filepath)
        print(f"Loaded full dataset with {len(full_data):,} rows from {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_filepath}. Please generate it first.")
        return

    # --- 2. Run Optuna Optimisation ---
    # Print device info once before starting
    print(f"\nStarting Optuna study on device: '{device}'")
    print(f"Tuning on a sample of {args.sample_size} rows for each of the {args.n_trials} trials.")

    study = optuna.create_study(directions=["maximize", "minimize"]) # Maximize AUC, Minimize Time
    study.optimize(
        lambda trial: objective(trial, full_data, tuning_config, args.sample_size, device),
        n_trials=args.n_trials,
    )

    # --- 3. Print Pareto Front Results ---
    print("\n--- Hyperparameter Tuning Finished ---")
    pareto_front = study.best_trials
    print(f"Found {len(pareto_front)} Pareto optimal trials.")
    if not pareto_front:
        print("No successful trials completed. Exiting.")
        return
        
    print("\nOptimal trials (Objective 0: AUC, Objective 1: Training Time):")
    for trial in pareto_front:
        print(f"  Trial {trial.number}:")
        print(f"    Values: AUC={trial.values[0]:.4f}, Time={trial.values[1]:.2f}s")
        print(f"    Params: {trial.params}")

    # --- 4. Generate and Save Tuning Visualisations ---
    print("\n--- Generating Tuning Visualisations ---")
    model_name = tuning_config["model_name"]
    output_plot_dir = os.path.join("reports", "figures", dataset_base_name, model_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"Saving plots to: {output_plot_dir}")
    
    # a) Pareto Front Plot
    plot_pareto_front(study, target_names=["AUC", "Training Time (s)"])
    save_path = os.path.join(output_plot_dir, "tuning_pareto_front.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() # Free memory
    print(f" - Saved Pareto front plot to: {save_path}")
    
    # b) Optimisation History Plot for AUC (Objective 0)
    plot_optimization_history(study, target=lambda t: t.values[0], target_name="AUC")
    save_path = os.path.join(output_plot_dir, "tuning_optimization_history_auc.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved AUC optimisation history plot to: {save_path}")
    
    # c) Optimisation History Plot for Training Time (Objective 1)
    plot_optimization_history(study, target=lambda t: t.values[1], target_name="Training Time (s)")
    save_path = os.path.join(output_plot_dir, "tuning_optimization_history_time.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved Training Time optimisation history plot to: {save_path}")
    
    # d) Hyperparameter Importance Plot (for AUC)
    plot_param_importances(study, target=lambda t: t.values[0], target_name="AUC")
    save_path = os.path.join(output_plot_dir, "tuning_param_importances_auc.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved hyperparameter importance plot (for AUC) to: {save_path}")

    # --- 5. User Selects the Best Trial ---
    selected_trial = None
    while not selected_trial:
        try:
            choice = input("\nEnter the number of the trial you want to use for the final config: ")
            trial_number = int(choice)
            # Find the trial by its number
            selected_trial = next((t for t in study.trials if t.number == trial_number), None)
            if selected_trial and selected_trial in pareto_front:
                print(f"You selected Trial {selected_trial.number}.")
            else:
                print("Invalid trial number or trial is not on the Pareto front. Please choose from the list above.")
                selected_trial = None
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid trial number.")

    # --- 6. Generate Final Configuration File ---
    print("\n--- Generating Final Configuration File ---")
    optimal_config_path = create_and_save_optimal_config(
        best_params=selected_trial.params,
        final_data_config_path=args.data_config,
        base_training_config_path=args.base_training_config,
        tuning_config=tuning_config,
    )

    # --- 7. Print Next Steps ---
    print("\n--- Next Step: Final Training on Full Dataset ---")
    print("To train your final model, run the following command:")
    print("\n" + "=" * 80)
    print(f"uv run run_training.py -dc {args.data_config} -tc {optimal_config_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
