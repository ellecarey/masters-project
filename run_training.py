import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data_generator_module.utils import create_filename_from_config

from src.data_generator_module import utils as data_utils
from src.training_module import utils as train_utils
from src.training_module.mlp_model import MLP
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model


def main():
    """
    Main function to run the MLP model training pipeline with train/val/test splits.
    It dynamically finds the dataset and saves the model based on the config.
    """
    parser = argparse.ArgumentParser(description="Run the model training pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yml",
        help="Path to the configuration YAML file for the experiment (default: config.yml)",
    )
    args = parser.parse_args()

    # Load config and set up reproducibility
    try:
        config_path = args.config
        config = data_utils.load_yaml_config(config_path)
        train_config = config["training_settings"]
        hyperparams = train_config["hyperparameters"]
        print("Successfully loaded configuration.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    global_seed = config["global_settings"]["random_seed"]
    train_utils.set_global_seed(global_seed)

    # Generate the unique name for this experiment to find the correct files
    experiment_name = create_filename_from_config(config)
    print(f"\nLooking for data for experiment: {experiment_name}")

    # Construct the full dynamic paths for the dataset and model
    output_data_dir = train_config["output_data_dir"]
    dataset_filepath = os.path.join(output_data_dir, f"{experiment_name}_dataset.csv")

    model_output_dir = train_config["model_output_dir"]
    model_filepath = os.path.join(model_output_dir, f"{experiment_name}_model.pt")

    # Load data using the dynamically generated path
    try:
        data = pd.read_csv(dataset_filepath)
        print(f"Loaded {len(data)} samples from {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{dataset_filepath}'.")
        print(
            "Please ensure you have run 'python run_generator.py' with the current config."
        )
        return

    # Prepare and split data
    target_column = train_config["target_column"]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    val_ratio = train_config["validation_set_ratio"]
    test_ratio = train_config["test_set_ratio"]

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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        criterion=criterion,
        optimiser=optimizer,
        epochs=hyperparams["epochs"],
    )

    # Final evaluation on the held-out test set
    print("\n--- Final Model Evaluation on Test Set ---")
    trained_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = trained_model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss (MSE): {avg_test_loss:.4f}")

    # Save the trained model using the dynamic filename
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    torch.save(trained_model.state_dict(), model_filepath)
    print(f"\nModel state dictionary saved to {model_filepath}")


if __name__ == "__main__":
    main()
