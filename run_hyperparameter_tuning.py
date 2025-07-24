import os
import argparse
from pathlib import Path
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
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model


# --- 1. Objective Function (with sampling and AUC) ---
def objective(trial, full_data, tuning_config, sample_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if trial.number == 0:
        print(f"Using device: {device}")
        print(f"Tuning on a sample of {sample_size} rows for each trial.")

    # --Create a stratified sample from the full dataset ---
    X_full = full_data.drop(columns=["target"])
    y_full = full_data["target"]
    
    # Use train_test_split to create a random, stratified sample
    # Use the trial number as a random seed for reproducibility of each trial
    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=sample_size, stratify=y_full, random_state=trial.number
    )

    # --- Suggest Hyperparameters  ---
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
    
    # --- Standard Training Setup on the SAMPLE ---
    data_utils.set_global_seed(trial.number) # Seed for weight initialization
    
    # Split the small sample into train/val for this trial
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=trial.number, stratify=y_sample
    )

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
    
    # --- Model Instantiation and Training ---
    model_name = tuning_config["model_name"]
    model_architecture_keys = ["hidden_size"] # Only include architecture params
    model_params = {key: hyperparams[key] for key in hyperparams if key in model_architecture_keys}
    model_params["input_size"] = X_sample.shape[1]
    model_params["output_size"] = 1
    
    model = get_model(model_name, model_params)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    
    trained_model, _ = train_model(
        model, train_loader, val_loader, criterion, optimiser, hyperparams["epochs"], device
    )

    # --- Evaluate on Validation Set and Return the AUC Score ---
    trained_model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = trained_model(features)
            scores = torch.sigmoid(outputs)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Return the AUC score to be maximized by Optuna
    auc = roc_auc_score(all_labels, all_scores)
    return auc

def plot_final_metrics(model, test_loader, device, experiment_name, base_filename, output_dir):
    """
    Evaluates the final model on the test set and plots the confusion matrix
    and ROC curve inside the specified output directory.
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

    all_preds = np.round(all_scores)

    # --- 1. Plot Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
    plt.title(f'Confusion Matrix for Best Model\n({experiment_name})')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    cm_save_path = os.path.join(output_dir, "confusion_matrix.pdf")
    plt.savefig(cm_save_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved confusion matrix to: {cm_save_path}")

    # --- 2. Plot ROC Curve ---
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
    roc_save_path = os.path.join(output_dir, "roc_curve.pdf")
    plt.savefig(roc_save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: {roc_save_path}")


def plot_training_history(history, experiment_name, base_filename, output_dir):
    """
    Plots the training and validation loss and accuracy over epochs,
    saving the result to the specified output directory.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Training History for Best Model\n({experiment_name})', fontsize=16)

    # --- Plot Loss ---
    ax1.plot(epochs, history['train_loss'], 'o-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')
    ax1.set_ylabel('Loss (BCE)')
    ax1.set_title('Model Loss Over Epochs')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot Accuracy ---
    ax2.plot(epochs, history['train_acc'], 'o-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'o-', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "training_history.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to: {save_path}")


def create_and_save_optimal_config(best_params, final_data_config_path, base_training_config_path):
    """
    Creates a new training config file populated with the best hyperparameters.
    """
    # Load the base training config to use as a template
    with open(base_training_config_path, 'r') as f:
        config_template = yaml.safe_load(f)

    # Update the hyperparameters with the best ones found by Optuna
    config_template['training_settings']['hyperparameters'] = best_params

    # Generate a descriptive name for the new config file
    final_data_config = data_utils.load_yaml_config(final_data_config_path)
    dataset_base_name = data_utils.create_filename_from_config(final_data_config)
    training_suffix = Path(base_training_config_path).stem
    new_config_filename = f"{dataset_base_name}_{training_suffix}_optimal.yml"

    # Define the save path
    save_dir = "configs/training/generated"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, new_config_filename)

    # Save the new configuration file
    with open(save_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, sort_keys=False)

    print(f"\nOptimal training configuration saved to: {save_path}")
    return save_path

# --- 2. The Main Tuning Pipeline (with sampling) ---

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Run hyperparameter tuning on a sample of a large dataset.")
    
    # --- Arguments ---
    parser.add_argument(
        "--data-config",
        "-dc",
        required=True,
        help="Path to the LARGE data config file.",
    )
    parser.add_argument(
        "--tuning-config",
        "-tc",
        required=True,
        help="Path to the tuning config file for the desired model.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples to use for each tuning trial.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of tuning trials to run."
    )
    args = parser.parse_args()

    # --- Load Full Dataset and Configs ---
    with open(args.tuning_config, "r") as f:
        tuning_config = yaml.safe_load(f)
    
    data_config = data_utils.load_yaml_config(args.data_config)
    dataset_base_name = data_utils.create_filename_from_config(data_config)
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")
    
    try:
        full_data = pd.read_csv(dataset_filepath)
        print(f"Loaded full dataset with {len(full_data):,} rows from {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_filepath}. Please generate it first.")
        return

    # --- Run Optuna Optimization on a Sample ---
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, full_data, tuning_config, args.sample_size),
        n_trials=args.n_trials,
    )

    # --- Print Best Results ---
    print("\n--- Hyperparameter Tuning Finished ---")
    best_trial = study.best_trial
    print(f"Best trial found at trial number: {best_trial.number}")
    print(f"Best AUC-ROC Score: {best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # --- Train Final Model on FULL Data ---
    print("\n--- Training Final Model on FULL Dataset with Best Hyperparameters ---")
    global_seed = data_config.get("global_settings", {}).get("random_seed", 42)
    data_utils.set_global_seed(global_seed)
    
    X = full_data.drop(columns=["target"])
    y = full_data["target"]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=global_seed, stratify=y
    )
    
    train_val_dataset = TabularDataset(X_train_val, y_train_val)
    test_dataset = TabularDataset(X_test, y_test)
    
    best_params = best_trial.params
    train_val_loader = DataLoader(dataset=train_val_dataset, batch_size=best_params["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=best_params["batch_size"], shuffle=False)
    
    model_name = tuning_config["model_name"]
    model_architecture_keys = ["hidden_size"]
    model_params = {key: best_params[key] for key in best_params if key in model_architecture_keys}
    model_params["input_size"] = X.shape[1]
    model_params["output_size"] = 1
    
    final_model = get_model(model_name, model_params)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
    
    trained_final_model, history = train_model(
        final_model, train_val_loader, train_val_loader, criterion, optimiser, best_params["epochs"], device=device
    )

    # --- Create Model-Specific Directory and Save All Plots ---
    model_name = tuning_config["model_name"]
    base_plot_dir = os.path.join("reports/figures", dataset_base_name)
    model_plot_dir = os.path.join(base_plot_dir, model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    
    experiment_name_for_title = f"{dataset_base_name}_{model_name}_tuning_best"
    
    plot_final_metrics(
        trained_final_model, test_loader, device, experiment_name_for_title, dataset_base_name, model_plot_dir
    )
    
    plot_training_history(
        history, experiment_name_for_title, dataset_base_name, model_plot_dir
    )
    
    print("\n--- Pipeline Complete ---")
    print(f"Plots for the {model_name} model have been saved in: {model_plot_dir}")



if __name__ == "__main__":
    main()