import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import json

from src.data_generator_module import utils as data_utils
from src.training_module import utils as train_utils
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model

def main():
    """
    Runs the final training pipeline using Path objects for robust path
    management. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--data-config", "-dc", type=str, required=True)
    parser.add_argument("--training-config", "-tc", type=str, required=True)
    args = parser.parse_args()

    # --- Load Configurations ---
    try:
        data_config = data_utils.load_yaml_config(args.data_config)
        training_config = data_utils.load_yaml_config(args.training_config)
        print(f"Loaded data config: {args.data_config}")
        print(f"Loaded training config: {args.training_config}")
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        return

    # --- Construct Final Experiment Name and File Paths ---
    train_settings = training_config["training_settings"]
    experiment_name = Path(args.training_config).stem
    print(f"\nRunning experiment: {experiment_name}")

    project_root = Path(data_utils.find_project_root())
    dataset_base_name = data_utils.create_filename_from_config(data_config)
    model_name = train_settings["model_name"]
    output_plot_dir = project_root / "reports" / "figures" / dataset_base_name / f"{model_name}_final"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = project_root / train_settings["model_output_dir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_filepath = model_output_dir / f"{experiment_name}_model.pt"

    # --- Load Data ---
    dataset_filepath = project_root / "data" / f"{dataset_base_name}_dataset.csv"
    try:
        data = pd.read_csv(dataset_filepath)
        print(f"Loaded {len(data)} samples from {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{dataset_filepath}'. Please generate it first.")
        return

    hyperparams = train_settings["hyperparameters"]
    global_seed = data_config.get("global_settings", {}).get("random_seed", 42)
    data_utils.set_global_seed(global_seed)

    # --- Data splitting and DataLoader ---
    target_column = train_settings["target_column"]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    val_ratio = train_settings["validation_set_ratio"]
    test_ratio = train_settings["test_set_ratio"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=global_seed, stratify=y
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=global_seed, stratify=y_temp
    )
    print(f"Data split into: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")
    
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    # --- Model initialisation, training, and scheduler creation ---
    ARCH_PARAMS = {"mlp_001": {"hidden_size", "output_size"}}
    valid_arch_keys = ARCH_PARAMS.get(model_name, set())
    model_params = {key: hyperparams[key] for key in hyperparams if key in valid_arch_keys}
    model_params["input_size"] = X_train.shape[1]

    model = get_model(model_name, model_params)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    scheduler_settings = train_settings.get("scheduler_settings", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode=scheduler_settings.get('mode', 'min'),
        factor=scheduler_settings.get('factor', 0.1),
        patience=scheduler_settings.get('patience', 10)
    )

    trained_model, history = train_model(
        model=model, train_loader=train_loader, validation_loader=val_loader,
        criterion=criterion, optimiser=optimiser, epochs=hyperparams["epochs"], device=device,
        scheduler=scheduler,
        verbose=True,
        early_stopping_enabled=False
    )

    # --- Final evaluation on the held-out test set ---
    print("\n--- Final Model Evaluation on Test Set ---")
    trained_model.eval()
    test_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = trained_model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            all_preds.extend(torch.round(torch.sigmoid(outputs)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    final_auc = roc_auc_score(all_labels, all_preds)

    print(f"Final Test Loss (BCE): {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {final_auc:.4f}")

    # --- Save final metrics to a JSON file ---
    final_metrics = {
        "Test Loss (BCE)": avg_test_loss, "Accuracy": accuracy, "F1-Score": f1,
        "Precision": precision, "Recall": recall, "AUC": final_auc
    }
    metrics_filepath = model_filepath.with_name(f"{model_filepath.stem}_metrics.json")
    with open(metrics_filepath, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Final metrics saved to: {metrics_filepath}")

    # --- Generating Final Evaluation Plots ---
    print("\n--- Generating Final Evaluation Plots ---")
    train_utils.plot_training_history(
        history=history,
        experiment_name=f"{model_name} on {dataset_base_name}",
        output_dir=output_plot_dir
    )
    train_utils.plot_final_metrics(
        model=trained_model, test_loader=test_loader, device=device,
        experiment_name=f"{model_name} on {dataset_base_name}",
        output_dir=output_plot_dir
    )

    torch.save(trained_model.state_dict(), model_filepath)
    print(f"\nModel state dictionary saved to {model_filepath}")

if __name__ == "__main__":
    main()
