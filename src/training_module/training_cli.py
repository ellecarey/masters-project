import torch
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn as nn
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data_generator_module import utils as data_utils
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.data_generator_module.utils import find_project_root 
from src.training_module import utils as train_utils
from src.training_module.models import get_model
from src.training_module.dataset import TabularDataset
from torch.utils.data import DataLoader
from src.training_module.trainer import train_model
from src.utils.filenames import experiment_name, metrics_filename, model_filename
from src.utils.plotting_helpers import generate_subtitle_from_config

TRAINING_SEED = 99
        
def train_single_config(data_config_path: str, training_config_path: str):
    """
    Train a model on a single data/training config pair.
    This function is restricted to only run on dedicated '_training' datasets.
    """
    apply_custom_plot_style()

    # --- Safeguard: Ensure this is a training dataset ---
    if "_training" not in Path(data_config_path).name:
        print("\nERROR: Invalid dataset for 'train-single'.")
        print("This command is exclusively for training on the dedicated '_training' dataset.")
        print(f"You provided: {Path(data_config_path).name}")
        print("\nPlease use a data config file with '_training_config.yml' in its name.")
        print("To evaluate your model on seeded datasets, use the 'evaluate-multiseed' command instead.")
        return

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
    full_base = data_utils.create_filename_from_config(data_config)
    
    exp_name = f"{full_base}_{model_name}_optimal"
    model_filepath_str = f"{full_base}_{model_name}_optimal_model.pt"
    metrics_filepath_str = f"{full_base}_{model_name}_optimal_metrics.json"

    print(f"\nRunning experiment: {exp_name}")

    project_root = Path(data_utils.find_project_root())
    model_output_dir = project_root / train_settings["model_output_dir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_filepath = model_output_dir / model_filepath_str
    metrics_filepath = model_output_dir / metrics_filepath_str

    # --- Load Data ---
    dataset_filepath = project_root / "data" / f"{full_base}_dataset.csv"
    try:
        data = pd.read_csv(dataset_filepath)
        print(f"Loaded {len(data)} samples from {dataset_filepath}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{dataset_filepath}'. Please generate it first.")
        return

    # --- Data splitting and DataLoader ---
    hyperparams = train_settings["hyperparameters"]
    global_seed = data_config.get("global_settings", {}).get("random_seed", 42)
    data_utils.set_global_seed(global_seed)
    
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
        patience=scheduler_settings.get('patience', 5)
    )
    
    early_stopping_settings = train_settings.get("early_stopping_settings", {})
    patience = early_stopping_settings.get("patience", 10)

    trained_model, history, _ = train_model(
        model=model, train_loader=train_loader, validation_loader=val_loader,
        criterion=criterion, optimiser=optimiser, epochs=hyperparams["epochs"], device=device,
        scheduler=scheduler,
        verbose=True,
        early_stopping_enabled=True,
        patience=patience
    )

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
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    final_auc = roc_auc_score(all_labels, all_preds)
    print(f"Final Test Loss (BCE): {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {final_auc:.4f}")

    final_metrics = {
        "Test Loss (BCE)": avg_test_loss, "Accuracy": accuracy, "F1-Score": f1,
        "Precision": precision, "Recall": recall, "AUC": final_auc
    }
    with open(metrics_filepath, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Final metrics saved to: {metrics_filepath}")

    plot_subtitle = generate_subtitle_from_config(data_config)

    print("\n--- Generating Final Evaluation Plots ---")
    train_utils.plot_training_history(
        history=history,
        experiment_name=exp_name,
        output_dir=Path("placeholder"),
        subtitle=plot_subtitle  # Pass the new subtitle
    )
    train_utils.plot_final_metrics(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        experiment_name=exp_name,
        output_dir=Path("placeholder"),
        subtitle=plot_subtitle  # Pass the new subtitle
    )
    
    torch.save(trained_model.state_dict(), model_filepath)
    print(f"\nModel state dictionary saved to {model_filepath}")


def evaluate_single_config(model_path: str, data_config_path: str, training_config_path: str):
    """
    Evaluate a pre-trained model on a single dataset's test split.
    This will overwrite any existing metrics file for this dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Evaluating model {model_path} on data from {data_config_path} ---")

    # --- Load Configurations ---
    try:
        data_config = data_utils.load_yaml_config(data_config_path)
        training_config = data_utils.load_yaml_config(training_config_path)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        return

    train_settings = training_config["training_settings"]
    model_name = train_settings["model_name"]
    hyperparams = train_settings["hyperparameters"]

    # --- Derive Experiment Name for Output Metrics 
    full_base = data_utils.create_filename_from_config(data_config)
    regex = r"(?P<base>.+?)(?:_(?P<pert>pert_[^_]+))?_seed(?P<seed>\d+)$"
    m = re.match(regex, full_base)
    if not m:
        # Fallback for the training file itself if it's evaluated by mistake
        if "_training" in full_base:
             print(f"Skipping evaluation on training file: {full_base}")
             return
        raise ValueError(f"Could not parse base/pert/seed from {full_base}")
    base, pert_tag, seed_str = m.group("base"), m.group("pert"), m.group("seed")
    seed = int(seed_str)
    
    project_root = Path(data_utils.find_project_root())
    model_output_dir = project_root / train_settings["model_output_dir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_filepath = model_output_dir / metrics_filename(
        base, model_name, seed=seed, perturbation_tag=pert_tag, optimized=False
    )
    print(f"Saving evaluation metrics to: {metrics_filepath}")

    # --- Load Data --- 
    dataset_filepath = project_root / "data" / f"{full_base}_dataset.csv"
    try:
        data = pd.read_csv(dataset_filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{dataset_filepath}'. Please generate it first.")
        return

    # --- Prepare Full Dataset for Evaluation 
    target_column = train_settings["target_column"]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    print(f"Using entire dataset of {len(X)} samples for evaluation.")
    
    evaluation_dataset = TabularDataset(X, y)
    evaluation_loader = DataLoader(
        dataset=evaluation_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False
    )
    
    # --- Model Initialisation and Loading State 
    ARCH_PARAMS = {"mlp_001": {"hidden_size", "output_size"}}
    valid_arch_keys = ARCH_PARAMS.get(model_name, set())
    model_params = {key: hyperparams[key] for key in hyperparams if key in valid_arch_keys}
    model_params["input_size"] = X.shape[1]
    
    model = get_model(model_name, model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # --- Evaluation on the Full Dataset ---
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    eval_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for features, labels in evaluation_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            all_preds.extend(torch.round(torch.sigmoid(outputs)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_eval_loss = eval_loss / len(evaluation_loader)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    final_auc = roc_auc_score(all_labels, all_preds)

    # --- Save Metrics --- 
    final_metrics = {
        "Test Loss (BCE)": avg_eval_loss, "Accuracy": accuracy, "F1-Score": f1,
        "Precision": precision, "Recall": recall, "AUC": final_auc
    }
    with open(metrics_filepath, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Evaluation metrics successfully saved to: {metrics_filepath}\n")

def evaluate_multi_seed(trained_model_path: str, data_config_base: str, optimal_config: str):
    """
    Evaluates a single pre-trained model over a multi-seed dataset family.
    This will ignore the dedicated training seed and skip datasets with existing metrics.
    """
    project_root = Path(data_utils.find_project_root())
    optimal_config_path = project_root / optimal_config
    
    # Load configs to get necessary parameters for constructing file paths
    try:
        training_config = data_utils.load_yaml_config(optimal_config_path)
    except FileNotFoundError as e:
        print(f"Error: Optimal training config file not found - {e}")
        return

    train_settings = training_config["training_settings"]
    model_name = train_settings["model_name"]
    model_output_dir = project_root / train_settings["model_output_dir"]

    base_data_config_path = project_root / data_config_base
    dataset_family_name = base_data_config_path.stem.replace('_config', '').split('_seed')[0]
    data_config_dir = project_root / "configs" / "data_generation"

    glob_pattern = f"{dataset_family_name}_seed*_config.yml"
    all_data_configs = sorted(list(data_config_dir.glob(glob_pattern)))
    # Exclude the training seed from evaluation
    evaluation_configs = [p for p in all_data_configs if "_training" not in p.name]

    if not evaluation_configs:
        print(f"Error: No evaluation data configs found for family '{dataset_family_name}' in '{data_config_dir}'")
        return

    print(f"\nFound {len(evaluation_configs)} datasets to evaluate using model '{trained_model_path}'.")
    evaluations_run = 0
    
    for data_config_path in evaluation_configs:
        # Load the specific data config to derive the output filename
        current_data_config = data_utils.load_yaml_config(data_config_path)
        full_base = data_utils.create_filename_from_config(current_data_config)

        # Replicate logic from evaluate_single_config to find the metrics file path
        regex = r"(?P<base>.+?)(?:_(?P<pert>pert_[^_]+))?_seed(?P<seed>\d+)$"
        m = re.match(regex, full_base)
        
        if not m:
            print(f"Warning: Could not parse base/pert/seed from {full_base}. Running evaluation without check.")
        else:
            base, pert_tag, seed_str = m.group("base"), m.group("pert"), m.group("seed")
            seed = int(seed_str)

            metrics_filepath = model_output_dir / metrics_filename(
                base, model_name, seed=seed, perturbation_tag=pert_tag, optimized=False
            )

            # Check if the metrics file already exists
            if metrics_filepath.exists():
                print(f"Skipping evaluation for {data_config_path.name}, metrics file already exists at '{metrics_filepath.name}'.")
                continue

        # If metrics file doesn't exist, run the evaluation
        evaluate_single_config(trained_model_path, str(data_config_path), str(optimal_config_path))
        evaluations_run += 1

    if evaluations_run > 0:
        print(f"\nRan {evaluations_run} new evaluation(s).")
    else:
        print("\nNo new evaluations were needed. All metrics files already exist.")

    print("\nMulti-dataset evaluation complete.")

    # --- Suggest next step: Aggregation ---
    print("\n" + "="*80)
    print("Next Step: Aggregate the results from all families")
    print("="*80 + "\n")
    print("Use the 'aggregate-all' command to summarize the performance metrics for both")
    print("the original and perturbed datasets.")
    
    aggregate_command = (
        f"uv run experiment_manager.py aggregate-all \\\n"
        f" --optimal-config {optimal_config}"
    )

    print(aggregate_command)
    print("\n" + "="*80)