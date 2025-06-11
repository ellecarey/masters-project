import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.training_module import utils as train_utils
from src.training_module.mlp_model import MLP
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model


def main():
    """
    Main function to run the MLP model training pipeline with train/val/test splits.
    """
    # Load config
    config = train_utils.load_yaml_config("config.yml")
    train_config = config["training_settings"]
    hyperparams = train_config["hyperparameters"]
    print("Successfully loaded configuration.")

    global_seed = config["global_settings"]["random_seed"]
    train_utils.set_global_seed(global_seed)

    # Load data
    try:
        data = pd.read_csv(train_config["data_path"])
        print(f"Loaded {len(data)} samples from {train_config['data_path']}")
    except FileNotFoundError:
        print("Data file not found. Please run 'python run_generator.py' first.")
        return

    target_column = train_config["target_column"]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Create Train, Validation, and Test Splits
    val_ratio = train_config["validation_set_ratio"]
    test_ratio = train_config["test_set_ratio"]

    # split data into training and temporary set (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=global_seed
    )

    # split the temporary set into validation and test sets
    # Adjust the split ratio for the remaining data
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=global_seed
    )

    print(
        f"Data split into: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples."
    )

    # 4. Create Datasets and DataLoaders
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

    # 5. Initialize model and training components
    input_size = X_train.shape[1]
    model = MLP(
        input_size=input_size,
        hidden_size=hyperparams["hidden_size"],
        output_size=hyperparams["output_size"],
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # 6. Train the model using the train and validation loaders
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=hyperparams["epochs"],
    )

    # 7. Final Evaluation on the Test Set (Held-out data)
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

    # 8. Save the trained model
    model_path = train_config["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel state dictionary saved to {model_path}")


if __name__ == "__main__":
    main()
