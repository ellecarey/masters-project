import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

from src.data_generator_module.utils import create_filename_from_config
from src.data_generator_module import utils as data_utils
from src.training_module import utils as train_utils
from src.training_module.mlp_model import MLP
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model


def main():
    """
    Runs the training pipeline using separate configuration files for
    data generation and training hyperparameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument(
        "--data-config",
        "-dc",
        type=str,
        required=True,
        help="Path to the data generation YAML config file.",
    )
    parser.add_argument(
        "--training-config",
        "-tc",
        type=str,
        required=True,
        help="Path to the training hyperparameters YAML config file (e.g., model-001.yml).",
    )
    args = parser.parse_args()

    # --- Load Both Configuration Files ---
    try:
        data_config = data_utils.load_yaml_config(args.data_config)
        training_config = data_utils.load_yaml_config(args.training_config)
        print(f"Loaded data config: {args.data_config}")
        print(f"Loaded training config: {args.training_config}")
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        return

    # --- Construct Final Experiment Name and File Paths ---
    # 1. Get the base name from the data configuration
    dataset_base_name = create_filename_from_config(data_config)

    # 2. Get the model suffix from the training configuration filename
    training_suffix = Path(args.training_config).stem  # e.g., "model-001"

    # 3. Combine them for the final, descriptive experiment name
    experiment_name = f"{dataset_base_name}_{training_suffix}"
    print(f"\nRunning experiment: {experiment_name}")

    # Define the input dataset path
    dataset_filepath = os.path.join("data", f"{dataset_base_name}_dataset.csv")

    # Define the output model path
    model_output_dir = training_config["training_settings"]["model_output_dir"]
    model_filepath = os.path.join(model_output_dir, f"{experiment_name}_model.pt")
    os.makedirs(model_output_dir, exist_ok=True)

    # --- Save a copy of the training config for full traceability ---
    output_config_dir = "configs/training/generated"
    os.makedirs(output_config_dir, exist_ok=True)
    new_config_path = os.path.join(output_config_dir, f"{experiment_name}_config.yml")
    with open(new_config_path, "w") as f:
        full_context = {
            "source_data_config": args.data_config,
            "source_training_config": args.training_config,
            "training_settings": training_config["training_settings"],
        }
        yaml.dump(full_context, f, default_flow_style=False)
    print(f"Saved full training context to: {new_config_path}")

    # --- Load Data ---
    try:
        data = pd.read_csv(dataset_filepath)
        print(f"Loaded {len(data)} samples from {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{dataset_filepath}'.")
        print("Please ensure you have generated this dataset first.")
        return

    # --- The rest of your training pipeline ---
    train_settings = training_config["training_settings"]
    hyperparams = train_settings["hyperparameters"]
    global_seed = data_config.get("global_settings", {}).get("random_seed", 42)
    data_utils.set_global_seed(global_seed)

    # Prepare and split data
    target_column = train_settings["target_column"]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    val_ratio = train_settings["validation_set_ratio"]
    test_ratio = train_settings["test_set_ratio"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=global_seed
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=global_seed
    )
    print(
        f"Data split into: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples."
    )

    # Create Datasets and DataLoaders
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    # Initialise and train the model
    model = MLP(
        input_size=X_train.shape[1],
        hidden_size=hyperparams["hidden_size"],
        output_size=hyperparams["output_size"],
    )
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        criterion=criterion,
        optimiser=optimiser,
        epochs=hyperparams["epochs"],
        device=device,
    )

    # Final evaluation on the held-out test set
    print("\n--- Final Model Evaluation on Test Set ---")
    trained_model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = trained_model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    # Calculate and print metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"Final Test Loss (BCE): {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save the trained model
    torch.save(trained_model.state_dict(), model_filepath)
    print(f"\nModel state dictionary saved to {model_filepath}")


if __name__ == "__main__":
    main()
