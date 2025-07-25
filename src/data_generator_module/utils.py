"""
Utility functions for the data generator module.
"""

import os
import shutil
from pathlib import Path
import yaml
import random
import numpy as np


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


def create_filename_from_config(config):
    """
    Creates a standardised filename base from a data generation config file.
    UPDATED: Now includes the random seed for better traceability.
    """
    dataset_settings = config.get("dataset_settings", {})
    class_config = config.get("create_feature_based_signal_noise_classification", {})
    global_settings = config.get("global_settings", {}) # Get global settings

    n_samples = dataset_settings.get("n_samples", 0)
    n_features = dataset_settings.get("n_initial_features", 0)
    
    # Extract feature types for the name
    feature_types = class_config.get("feature_types", {})
    continuous_count = sum(1 for ft in feature_types.values() if ft == "continuous")
    discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")

    # Calculate average separation for the name
    signal_features = class_config.get("signal_features", {})
    noise_features = class_config.get("noise_features", {})
    separations = [
        abs(signal_features[f]['mean'] - noise_features.get(f, {}).get('mean', 0))
        for f in signal_features if f in noise_features
    ]
    avg_separation = sum(separations) / len(separations) if separations else 0.0
    
    # --- NEW: Get the random seed ---
    random_seed = global_settings.get("random_seed", 42)

    # Construct the filename
    name_parts = [
        f"n{n_samples}",
        f"f_init{n_features}",
        f"cont{continuous_count}",
        f"disc{discrete_count}",
        f"sep{str(avg_separation).replace('.', 'p')}",
        f"seed{random_seed}"
    ]

    return "_".join(name_parts)


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
        n_added = config.get("add_features", {}).get("n_new_features", 0)
        total_features = n_initial + n_added

        # Perturbation description
        pert_settings = config.get("perturbation", {})
        pert_type = pert_settings.get("perturbation_type", "none")
        if pert_type != "none":
            pert_scale = pert_settings.get("scale", 0)
            pert_desc = f"Perturbation: {pert_type.capitalize()} (Scale: {pert_scale})"
        else:
            pert_desc = "No Perturbations"

        # Target variable description - ADD THIS CHECK
        if "create_feature_based_signal_noise_classification" in config:
            target_desc = "Target: Feature-based Classification"
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


def set_global_seed(seed: int):
    """
    Sets the random seed for Python, NumPy to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Global random seed set to {seed}")
