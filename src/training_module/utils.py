import os
import shutil
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.calibration import calibration_curve
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

def plot_training_history(history: dict, experiment_name: str, output_dir: str, subtitle: str = None):
    """
    Plots training and validation loss and accuracy from a history dictionary.
    
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plotting Loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting Accuracy
    ax2.plot(history['train_acc_epoch_end'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    if subtitle:
        fig.suptitle(subtitle)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()

    save_path = os.path.join(output_dir, f"{experiment_name}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved training history plot to: {save_path}")



def plot_final_metrics(model, test_loader, device, model_name: str, trial_number: int, output_dir: str, subtitle: str = None):
    """
    Generates and saves ROC curve and confusion matrix plots with consistent naming.
    """
    model.to(device)
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            scores = torch.sigmoid(outputs).cpu().numpy()
            all_scores.extend(scores.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if subtitle:
        plt.suptitle(subtitle, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    roc_save_path = os.path.join(output_dir, f"{model_name}_roc_curve_trial{trial_number}.pdf")
    plt.savefig(roc_save_path, bbox_inches='tight')
    plt.close()

    # --- Confusion Matrix ---
    predictions = [1 if score > 0.5 else 0 for score in all_scores]
    cm = confusion_matrix(all_labels, predictions)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if subtitle:
        plt.suptitle(subtitle, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    cm_save_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_trial{trial_number}.pdf")
    plt.savefig(cm_save_path, bbox_inches='tight')
    plt.close()

def plot_combined_training_histories(candidate_info: list, output_dir: str, model_name: str, subtitle: str = None):
    """
    Plots training histories with fully adaptive, padded axes for maximum readability.
    Each subplot scales independently to show its own training dynamics clearly.
    """
    if not candidate_info:
        return

    num_candidates = len(candidate_info)
    # Each subplot will have fully independent axes for maximum clarity.
    fig, axes = plt.subplots(
        num_candidates, 2,
        figsize=(16, 5 * num_candidates),
        sharex=False, # Independent x-axes
        sharey=False, # Independent y-axes
        squeeze=False
    )

    for i, info in enumerate(candidate_info):
        history = info.get('history', {})
        ax_loss = axes[i, 0]
        ax_acc = axes[i, 1]

        # --- Plot Titles (working correctly) ---
        rank = info.get('rank', 'N/A')
        trial_number = info.get('trial_number', 'N/A')
        training_time = info.get('training_time', 'N/A')
        final_val_loss = history.get('val_loss', [float('nan')])[-1]
        final_val_acc = history.get('val_acc', [float('nan')])[-1]
        candidate_title = (
            f"Candidate {rank} (Trial #{trial_number}) | Time: {training_time}s\n"
            f"Final Val Loss: {final_val_loss:.4f} | Final Val Acc: {final_val_acc:.4f}"
        )
        ax_loss.set_title(candidate_title, loc='left', pad=10)
        ax_acc.set_title("Accuracy Curves", loc='left', pad=10)
        ax_loss.set_ylabel("Loss")
        ax_acc.set_ylabel("Accuracy")

        # --- Plotting Data ---
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        ax_loss.plot(train_loss, label='Train Loss', color='C0')
        ax_loss.plot(val_loss, label='Validation Loss', color='C1')
        ax_loss.legend()
        ax_loss.grid(True)

        train_acc = history.get('train_acc_epoch_end', [])
        val_acc = history.get('val_acc', [])
        ax_acc.plot(train_acc, label='Train Accuracy', color='C0')
        ax_acc.plot(val_acc, label='Validation Accuracy', color='C1')
        ax_acc.legend()
        ax_acc.grid(True)

        # --- MODIFICATION: Adaptive and Padded Axis Limits ---

        # 1. X-Axis Limits (for both loss and accuracy plots)
        num_epochs = len(train_loss)
        if num_epochs > 1:
            # Add a small buffer on both sides of the x-axis
            x_buffer = max(1, num_epochs * 0.05)
            ax_loss.set_xlim(-x_buffer, num_epochs - 1 + x_buffer)
            ax_acc.set_xlim(-x_buffer, num_epochs - 1 + x_buffer)
        else:
            ax_loss.set_xlim(-0.5, 1.5)
            ax_acc.set_xlim(-0.5, 1.5)

        # --- Integer x-axis ticks for epoch labels ---
        if num_epochs > 0:
            # Set x-axis ticks to be integers only
            ax_loss.set_xticks(range(0, num_epochs, max(1, num_epochs // 10)))
            ax_acc.set_xticks(range(0, num_epochs, max(1, num_epochs // 10)))
            
            # Force integer formatting for x-axis labels
            ax_loss.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            ax_acc.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

        # 2. Y-Axis Limits for Loss Plot
        all_losses = train_loss + val_loss
        if all_losses:
            y_max = np.max(all_losses)
            # Start at 0 and add 5% padding to the top
            ax_loss.set_ylim(bottom=0, top=y_max * 1.1)

        all_accs = train_acc + val_acc
        if all_accs:
            y_min, y_max = np.min(all_accs), np.max(all_accs)
            y_range = y_max - y_min
            
            # Same logic as loss: start from reasonable minimum, scale to data + padding
            if y_range > 0:
                # Use the minimum accuracy as the base (like loss uses 0)
                # Add small buffer below and 10% padding above (same as loss)
                padding_below = min(0.02, y_range * 0.1)  # Small buffer below, max 2%
                lower_bound = max(0.0, y_min - padding_below)
                upper_bound = y_max * 1.1  # Same 10% padding as loss plots
                ax_acc.set_ylim(lower_bound, min(upper_bound, 1.01))  # Cap at reasonable max
            else:
                # Fallback for flat lines
                center = y_min
                ax_acc.set_ylim(max(0.0, center - 0.02), min(center + 0.02, 1.05))
        else:
            ax_acc.set_ylim(0.0, 1.05)  # Standard fallback

    # --- Final Touches (unchanged) ---
    if num_candidates > 0:
        axes[-1, 0].set_xlabel("Epoch")
        axes[-1, 1].set_xlabel("Epoch")

    if subtitle:
        main_title = "Combined Training History for Top Candidates"
        fig.suptitle(f"{main_title}\n{subtitle}")
    else:
        fig.suptitle("Combined Training History for Top Candidates")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, f"{model_name}_combined_training_history.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined training history plot with fully adaptive axes to: {save_path}")


