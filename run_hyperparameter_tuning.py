import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, auc


from src.data_generator_module import utils as data_utils
from src.training_module.mlp_model import MLP
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model


# --- 1. The Objective Function ---
# This function defines a single training and validation trial.
def objective(trial, data_config_path, tuning_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if trial.number == 0:  # Print only for the first trial
        print(f"Using device: {device}")
    try:
        data_config = data_utils.load_yaml_config(data_config_path)
        dataset_base_name = data_utils.create_filename_from_config(data_config)
        dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")
        data = pd.read_csv(dataset_filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found for config {data_config_path}")
        return float("inf")

    # --- Suggest Hyperparameters for this Trial (Corrected Logic) ---
    search_space = tuning_config["search_space"]
    hyperparams = {}
    for param_name, params in search_space.items():
        param_type = params["type"]
        # Create a clean copy of params, removing 'type'
        suggestion_params = {k: v for k, v in params.items() if k != "type"}

        if param_type == "int":
            hyperparams[param_name] = trial.suggest_int(param_name, **suggestion_params)
        elif param_type == "float":
            hyperparams[param_name] = trial.suggest_float(
                param_name, **suggestion_params
            )
        elif param_type == "categorical":
            # suggest_categorical expects choices as a positional argument
            hyperparams[param_name] = trial.suggest_categorical(
                param_name, **suggestion_params
            )

    # Add the fixed output_size parameter
    hyperparams["output_size"] = 1

    # --- Standard Training Setup (no changes needed below this line) ---
    global_seed = data_config.get("global_settings", {}).get("random_seed", 42)
    data_utils.set_global_seed(global_seed)

    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=global_seed
    )

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    model = MLP(
        input_size=X.shape[1],
        hidden_size=hyperparams["hidden_size"],
        output_size=hyperparams["output_size"],
    )
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # --- Train and Validate the Model ---
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimiser,
        hyperparams["epochs"],
        device=device,
    )

    # --- Evaluate on Validation Set and Return the Metric to Optimize ---
    trained_model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            # Get the raw output scores from the model
            outputs = trained_model(features)
            # Apply sigmoid to get probabilities, but do NOT round
            scores = torch.sigmoid(outputs)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Maximize the AUC-ROC score
    auc = roc_auc_score(all_labels, all_scores)
    return auc


    # # --- Evaluate on Validation Set and Return the Metric to Optimize ---
    # trained_model.eval()
    # all_preds = []
    # all_labels = []
    # with torch.no_grad():
    #     for features, labels in val_loader:
    #         # Move the validation batch to the correct device
    #         features = features.to(device)
    
    #         outputs = trained_model(features)
    #         preds = torch.round(torch.sigmoid(outputs))
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(labels.cpu().numpy())

    # # maximize the F1-score
    # f1 = f1_score(all_labels, all_preds)
    # return f1

def plot_final_metrics(model, test_loader, device, experiment_name):
    """
    Evaluates the final model on the test set and plots the confusion matrix
    and ROC curve.
    """
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            scores = torch.sigmoid(outputs)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Get binary predictions for the confusion matrix
    all_preds = np.round(all_scores)

    # --- 1. Plot and Save Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
    plt.title(f'Confusion Matrix for Best Model\n({experiment_name})')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    cm_save_path = f"reports/figures/{experiment_name}_confusion_matrix.pdf"
    plt.savefig(cm_save_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved confusion matrix to: {cm_save_path}")

    # --- 2. Plot and Save ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC)\n({experiment_name})')
    plt.legend(loc="lower right")
    roc_save_path = f"reports/figures/{experiment_name}_roc_curve.pdf"
    plt.savefig(roc_save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: {roc_save_path}")



# --- 2. The Main Tuning Pipeline ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning.")
    parser.add_argument(
        "--data-config",
        "-dc",
        required=True,
        help="Path to the data config file defining the dataset to tune on.",
    )
    parser.add_argument(
        "--tuning-config",
        "-tc",
        required=True,
        help="Path to the tuning config file defining the hyperparameter search space.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of tuning trials to run."
    )
    args = parser.parse_args()

    # Load the tuning configuration
    with open(args.tuning_config, "r") as f:
        tuning_config = yaml.safe_load(f)

    # --- Create an Optuna Study ---
    # specify 'maximize' - highest F1-score.
    study = optuna.create_study(direction="maximize")

    # --- Run the Optimization ---
    # pass the objective function and other fixed arguments using a lambda function.
    study.optimize(
        lambda trial: objective(trial, args.data_config, tuning_config),
        n_trials=args.n_trials,
    )

    # --- Print the Best Results ---
    print("\n--- Hyperparameter Tuning Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial found at trial number: {best_trial.number}")
    print(f"Best AUC-ROC Score: {best_trial.value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print("\n--- Training Final Model with Best Hyperparameters ---")
    
    data_config = data_utils.load_yaml_config(args.data_config)
    global_seed = data_config.get("global_settings", {}).get("random_seed", 42)
    
    # 1. Load data and create splits
    dataset_base_name = data_utils.create_filename_from_config(data_config)
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")
    data = pd.read_csv(dataset_filepath)

    X = data.drop(columns=["target"])
    y = data["target"]
    
    # Create the same splits as run_training.py to get the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=global_seed
    )

    train_val_dataset = TabularDataset(X_train_val, y_train_val)
    test_dataset = TabularDataset(X_test, y_test)
    
    best_params = best_trial.params
    train_val_loader = DataLoader(
        dataset=train_val_dataset, batch_size=best_params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=best_params["batch_size"], shuffle=False
    )

    # 2. Re-initialize and train the final model
    data_utils.set_global_seed(global_seed)
    final_model = MLP(
        input_size=X.shape[1],
        hidden_size=best_params["hidden_size"],
        output_size=1
    )
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])

    # Train on the combined training and validation data
    trained_final_model = train_model(
        final_model,
        train_val_loader,
        train_val_loader, # Use the same loader for val since it's just for progress
        criterion,
        optimiser,
        best_params["epochs"],
        device=device,
    )

    # 3. Generate and save plots using the held-out test set
    experiment_name = f"{dataset_base_name}_tuning_best"
    plot_final_metrics(trained_final_model, test_loader, device, experiment_name)

if __name__ == "__main__":
    main()
