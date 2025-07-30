import os
import shutil
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.utils.report_paths import artefact_path, experiment_family_path, extract_family_base
import re
from src.utils.plotting_helpers import format_plot_title, apply_decimal_formatters

def find_project_root():
    """Find the project root by searching upwards for a marker file."""
    # Start from the directory of this file (__file__).
    current_path = Path(__file__).resolve()

    # Define project root markers.
    markers = [".git", "pyproject.toml", "README.md", "run_data_generator.py"]

    for parent in current_path.parents:
        # Check if any marker file exists in the current parent directory.
        if any((parent / marker).exists() for marker in markers):
            # If a marker is found, we have found the project root.
            print(f"Project root found at: {parent}")
            return str(parent)

    # --- FALLBACK ---
    # Last resort if no markers are found
    # Assumes a fixed structure: utils.py -> generator_package -> src -> masters-project
    fallback_path = current_path.parent.parent.parent
    print(
        f"Warning: No project root marker found. Using fallback path: {fallback_path}"
    )
    return str(fallback_path)


def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_project_paths():
    """Gets a dictionary of important project paths."""
    project_root = find_project_root()
    paths = {
        "project_root": project_root,
        "data_path": os.path.join(project_root, "data"),
        "figures_path": os.path.join(project_root, "reports", "figures"),
        "notebooks_path": os.path.join(project_root, "notebooks"),
        "src_path": os.path.join(project_root, "src"),
    }
    return paths


def create_filename_from_config(config: dict) -> str:
    """Create a unique filename based on configuration parameters."""

    # Extract dataset settings
    dataset_settings = config.get("dataset_settings", {})
    n_samples = dataset_settings.get("n_samples", 1000)
    n_initial_features = dataset_settings.get("n_initial_features", 5)

    # Extract feature type distribution
    feature_generation = config.get("feature_generation", {})
    feature_types = feature_generation.get("feature_types", {})

    continuous_count = sum(1 for ft in feature_types.values() if ft == "continuous")
    discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")

    # Extract additional features (removed since not needed for signal/noise)
    n_new_features = 0

    # Extract perturbation settings (removed since not needed for signal/noise)
    perturbation_type = "none"
    perturbation_scale = 0

    # Extract target settings - handle both signal/noise and regular targets
    if "create_signal_noise_target" in config:
        signal_config = config["create_signal_noise_target"]
        function_type = f"signal-{signal_config.get('signal_function', 'linear')}"
        signal_ratio = signal_config.get("signal_ratio", 0.5)
        noise_level = signal_config.get("noise_level", 0.1)
        filename_parts = [
            f"n{n_samples}",
            f"f_init{n_initial_features}",
            f"cont{continuous_count}",
            f"disc{discrete_count}",
            f"add{n_new_features}",
            f"pert-{perturbation_type}",
            f"scl{str(perturbation_scale).replace('.', 'p')}",
            f"func-{function_type}",
            f"sig-ratio{str(signal_ratio).replace('.', 'p')}",
            f"noise{str(noise_level).replace('.', 'p')}",
        ]
    else:
        target = config.get("create_target", {})
        function_type = target.get("function_type", "linear")
        noise_level = target.get("noise_level", 0.1)

        if function_type == "signal_noise":
            signal_ratio = target.get("signal_ratio", 0.5)
            filename_parts = [
                f"n{n_samples}",
                f"f_init{n_initial_features}",
                f"cont{continuous_count}",
                f"disc{discrete_count}",
                f"add{n_new_features}",
                f"pert-{perturbation_type}",
                f"scl{str(perturbation_scale).replace('.', 'p')}",
                f"func-{function_type}",
                f"sig-ratio{str(signal_ratio).replace('.', 'p')}",
                f"noise{str(noise_level).replace('.', 'p')}",
            ]
        else:
            # Original logic for regression targets
            filename_parts = [
                f"n{n_samples}",
                f"f_init{n_initial_features}",
                f"cont{continuous_count}",
                f"disc{discrete_count}",
                f"add{n_new_features}",
                f"pert-{perturbation_type}",
                f"scl{str(perturbation_scale).replace('.', 'p')}",
                f"func-{function_type}",
                f"noise{str(noise_level).replace('.', 'p')}",
            ]

    return "_".join(filename_parts)


def create_plot_title_from_config(config: dict) -> tuple[str, str]:
    """
    Generates a human-readable title and subtitle for plots from the config.
    """
    try:
        # Main Title
        main_title = "Distribution of Generated Features"

        # Subtitle Components
        ds_settings = config.get("dataset_settings", {})
        n_samples = ds_settings.get("n_samples", "N/A")

        # Calculate total features
        n_initial = ds_settings.get("n_initial_features", 0)
        n_added = 0  # Removed add_features for signal/noise approach
        total_features = n_initial + n_added

        # Perturbation description (removed for signal/noise)
        pert_desc = "No Perturbations"

        # Target variable description - handle signal/noise
        if "create_signal_noise_target" in config:
            signal_config = config["create_signal_noise_target"]
            signal_function = signal_config.get("signal_function", "linear")
            signal_ratio = signal_config.get("signal_ratio", 0.5)
            target_desc = f"Target: Signal/Noise Classification ({signal_function}, {signal_ratio:.1%} signal)"
        else:
            func_type = config.get("create_target", {}).get("function_type", "N/A")
            if func_type == "signal_noise":
                target_desc = "Target: Signal/Noise Classification"
            else:
                target_desc = f"Target: {func_type.capitalize()} Relationship"

        # Assemble the subtitle
        subtitle = (
            f"Dataset: {n_samples:,} Samples, {total_features} Features | "
            f"{pert_desc} | {target_desc}"
        )

        return main_title, subtitle

    except Exception:
        # Fallback if the config structure is unexpected
        return "Feature Distribution", "Configuration details unavailable"


def rename_config_file(original_config_path, experiment_name):
    """
    Rename the configuration file to match the generated dataset name.
    """
    config_path = Path(original_config_path)
    config_dir = config_path.parent
    config_extension = config_path.suffix

    # Create new filename
    new_config_name = f"{experiment_name}_config{config_extension}"
    new_config_path = config_dir / new_config_name

    try:
        # Rename the file
        shutil.move(str(config_path), str(new_config_path))
        print(f"Configuration file renamed: {config_path.name} → {new_config_name}")
        return str(new_config_path)
    except Exception as e:
        print(f"Warning: Could not rename config file: {e}")
        return str(config_path)

def plot_training_history(history, experiment_name, output_dir):
    """
    Plots the training and validation loss and accuracy over epochs,
    saving the result to the specified output directory.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # The figsize and suptitle fontsize arguments are removed to inherit from rcParams.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    title = format_plot_title("Training History for", experiment_name)
    fig.suptitle(title)

    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'o-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')
    ax1.set_ylabel('Loss (BCE)')
    ax1.set_title('Model Loss Over Epochs')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'o-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'o-', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    apply_decimal_formatters(ax1, precision=3)
    apply_decimal_formatters(ax2, precision=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    
    subfolder = re.sub(r'_mlp_.*$', '', experiment_name)
    
    save_path = experiment_family_path(
        full_experiment_name=experiment_name,
        art_type="figure",
        subfolder=subfolder,
        filename="training_history.pdf"
    )

    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to: {save_path}")


def plot_final_metrics(model, test_loader, device, experiment_name, output_dir):
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
    
    subfolder = re.sub(r'_mlp_.*$', '', experiment_name)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
    cm_title = format_plot_title("Confusion Matrix for", experiment_name, sub_width=60)
    plt.title(cm_title)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    cm_save_path = experiment_family_path(
        full_experiment_name=experiment_name,
        art_type="figure",
        subfolder=subfolder,
        filename="confusion_matrix.pdf"
    )
    plt.savefig(cm_save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {cm_save_path}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_title = format_plot_title("Receiver Operating Characteristic (ROC)", experiment_name, sub_width=60)
    plt.title(roc_title)
    plt.legend(loc="lower right")

    roc_save_path = experiment_family_path(
        full_experiment_name=experiment_name,
        art_type="figure",
        subfolder=subfolder,
        filename="roc_curve.pdf"
    )
    plt.savefig(roc_save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: {roc_save_path}")
