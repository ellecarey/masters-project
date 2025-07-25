import os
import argparse
from pathlib import Path
import pandas as pd
import time
import torch
import torch.nn as nn
import yaml
import optuna
import copy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from src.data_generator_module import utils as data_utils
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model

def objective(trial, full_data, tuning_config, sample_fraction, device):
    """
    The objective function for Optuna to optimise.
    It trains a model on a sample of the data and returns performance (AUC) and training time.
    """
    X_full = full_data.drop(columns=["target"])
    y_full = full_data["target"]

    if sample_fraction >= 1.0:
        X_sample, y_sample = X_full, y_full
    else:
        X_sample, _, y_sample, _ = train_test_split(
            X_full, y_full, train_size=sample_fraction, stratify=y_full, random_state=trial.number
        )

    # Suggest tunable hyperparameters from the search space
    search_space = tuning_config["search_space"]
    hyperparams = {}
    for param_name, params_config in search_space.items():
        # Create a shallow copy to prevent modifying the original config dict
        params = copy.copy(params_config)
        param_type = params.pop("type")

        if param_type == "int":
            if "step" in params:
                hyperparams[param_name] = trial.suggest_int(
                    param_name, params["low"], params["high"], step=params["step"]
                )
            else:
                hyperparams[param_name] = trial.suggest_int(param_name, **params)
        elif param_type == "float":
            hyperparams[param_name] = trial.suggest_float(param_name, **params)
        elif param_type == "categorical":
            hyperparams[param_name] = trial.suggest_categorical(param_name, **params)

    # Set up model parameters
    model_name = tuning_config["model_name"]
    ARCH_PARAMS = {
        "MLP": {"hidden_size"},
        "LogisticRegression": {}
    }
    valid_arch_keys = ARCH_PARAMS.get(model_name, set())
    model_params = {key: hyperparams[key] for key in hyperparams if key in valid_arch_keys}
    
    model_params["input_size"] = X_sample.shape[1]
    model_params["output_size"] = 1
    
    model = get_model(model_name, model_params)

    # Set up data loaders and training components
    data_utils.set_global_seed(trial.number)
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=trial.number, stratify=y_sample
    )
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Train and evaluate the model
    start_time = time.time()
    trained_model, _ = train_model(
        model, train_loader, val_loader, criterion, optimiser, hyperparams["epochs"], device
    )
    training_time = time.time() - start_time

    trained_model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, outputs = features.to(device), trained_model(features.to(device))
            all_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_scores)
    return auc, training_time

def main():
    """
    Main function to run the distributed hyperparameter tuning pipeline.
    This script's only purpose is to run trials and populate the study database.
    """
    parser = argparse.ArgumentParser(description="Run distributed hyperparameter tuning.")
    parser.add_argument("--data-config", "-dc", required=True, help="Path to the data config file to sample from.")
    parser.add_argument("--tuning-config", "-tc", required=True, help="Path to the tuning config file with the search space.")
    parser.add_argument("--base-training-config", "-btc", required=True, help="Path to the base training config template.")
    parser.add_argument("--sample-fraction", type=float, default=0.8, help="Fraction of the dataset to use for each tuning trial.")
    parser.add_argument("--n-trials", type=int, default=50, help="Total number of tuning trials to run across all workers.")
    args = parser.parse_args()

    # --- 1. Load Full Dataset and Configs ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuning_config = data_utils.load_yaml_config(args.tuning_config)
    data_config = data_utils.load_yaml_config(args.data_config)
    dataset_base_name = data_utils.create_filename_from_config(data_config)
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")

    try:
        full_data = pd.read_csv(dataset_filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_filepath}. Please generate it first.")
        return

    # --- 2. Define and Set Up a Resilient Distributed Study ---
    storage_name = f"sqlite:///reports/{dataset_base_name}_tuning.db"
    model_name_suffix = Path(args.base_training_config).stem
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["maximize", "minimize"],
        load_if_exists=True
    )
    print(f"\nStarting or joining Optuna study '{study_name}' on device: '{device}'")
    
    # --- 3. Run Optuna Optimisation ---
    study.optimize(
        lambda trial: objective(trial, full_data, tuning_config, args.sample_fraction, device),
        callbacks=[MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE,))]
    )
    
    print(f"\nWorker has finished its trials for study '{study_name}'.")

if __name__ == "__main__":
    main()
