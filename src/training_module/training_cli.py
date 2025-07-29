import torch
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn as nn
import re
from src.data_generator_module import utils as data_utils
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.training_module import utils as train_utils
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from src.training_module.trainer import train_model

from src.utils.filenames import experiment_name, metrics_filename, model_filename

def train_single_config(data_config_path: str, training_config_path: str):
    """
    Train a model on a single data/training config pair.
    """
    apply_custom_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Configurations ---
    try:
        data_config = data_utils.load_yaml_config(data_config_path)
        training_config = data_utils.load_yaml_config(training_config_path)
        print(f"Loaded data config: {data_config_path}")
        print(f"Loaded training config: {training_config_path}")
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        return

    train_settings = training_config["training_settings"]
    model_name = train_settings["model_name"]

    # --- Generate the canonical base, perturbation tag, and seed ---
    full_base = data_utils.create_filename_from_config(data_config)
    # This regex strips optional _pert_... and always strips _seedN at the end
    regex = r"(?P<base>.+?)(?:_(?P<pert>pert_[^_]+))?_seed(?P<seed>\d+)$"
    m = re.match(regex, full_base)
    if not m:
        raise ValueError(f"Could not parse base/pert/seed from {full_base}")
    base = m.group("base")
    pert_tag = m.group("pert")
    seed = int(m.group("seed"))

    exp_name = experiment_name(base, model_name, seed=seed, perturbation_tag=pert_tag)
    print(f"\nRunning experiment: {exp_name}")

    project_root = Path(data_utils.find_project_root())
    output_plot_dir = Path("placeholder")
    model_output_dir = project_root / train_settings["model_output_dir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_filepath = model_output_dir / model_filename(base, model_name, seed=seed, perturbation_tag=pert_tag)
    metrics_filepath = model_output_dir / metrics_filename(base, model_name, seed=seed, perturbation_tag=pert_tag)

    # --- Load Data ---
    dataset_filepath = project_root / "data" / f"{full_base}_dataset.csv"
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
    from torch.utils.data import DataLoader
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
        patience=scheduler_settings.get('patience', 5)
    )

    early_stopping_settings = train_settings.get("early_stopping_settings", {})
    patience = early_stopping_settings.get("patience", 10)

    trained_model, history = train_model(
        model=model, train_loader=train_loader, validation_loader=val_loader,
        criterion=criterion, optimiser=optimiser, epochs=hyperparams["epochs"], device=device,
        scheduler=scheduler,
        verbose=True,
        early_stopping_enabled=True,
        patience=patience
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
    with open(metrics_filepath, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Final metrics saved to: {metrics_filepath}")

    # --- Generating Final Evaluation Plots ---
    print("\n--- Generating Final Evaluation Plots ---")
    train_utils.plot_training_history(
        history=history,
        experiment_name=exp_name,  # Use clean experiment name
        output_dir=output_plot_dir
    )
    
    train_utils.plot_final_metrics(
        model=trained_model, test_loader=test_loader, device=device,
        experiment_name=exp_name,  # Use clean experiment name  
        output_dir=output_plot_dir
    )

    torch.save(trained_model.state_dict(), model_filepath)
    print(f"\nModel state dictionary saved to {model_filepath}")




def train_multi_seed(data_config_base: str, optimal_config: str):
    """
    Train the same optimal config over a multi-seed dataset family.
    """
    from pathlib import Path
    from src.training_module.training_cli import train_single_config  
    from src.data_generator_module.utils import find_project_root

    project_root = Path(find_project_root())
    base_data_config_path = project_root / data_config_base
    optimal_config_path = project_root / optimal_config
    dataset_family_name = base_data_config_path.stem.replace('_config', '').split('_seed')[0]
    data_config_dir = project_root / "configs" / "data_generation"
    all_data_configs = sorted(list(data_config_dir.glob(f"{dataset_family_name}_seed*_config.yml")))
    for data_config in all_data_configs:
        print(f"Training on: {data_config.name}")
        train_single_config(str(data_config), str(optimal_config_path))
    print("\nMulti-seed training complete.")
