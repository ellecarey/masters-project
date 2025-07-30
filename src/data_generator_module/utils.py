import os
import shutil
from pathlib import Path
import yaml
import random
import numpy as np
import re

TRAINING_SEED = 99

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
    """
    dataset_settings = config.get("dataset_settings", {})
    class_config = config.get("create_feature_based_signal_noise_classification", {})
    global_settings = config.get("global_settings", {})
    perturbations = config.get("perturbation_settings", [])

    n_samples = dataset_settings.get("n_samples", 0)
    n_features = dataset_settings.get("n_initial_features", 0)
    
    feature_types = class_config.get("feature_types", {})
    continuous_count = sum(1 for ft in feature_types.values() if ft == "continuous")
    discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")
    
    signal_features = class_config.get("signal_features", {})
    noise_features = class_config.get("noise_features", {})
    separations = [
        abs(signal_features[f]['mean'] - noise_features.get(f, {}).get('mean', 0))
        for f in signal_features if f in noise_features
    ]
    avg_separation = sum(separations) / len(separations) if separations else 0.0
    random_seed = global_settings.get("random_seed", 42)

    name_parts = [
        f"n{n_samples}",
        f"f_init{n_features}",
        f"cont{continuous_count}",
        f"disc{discrete_count}",
        f"sep{str(avg_separation).replace('.', 'p')}"
    ]

    if perturbations:
        pert_str_parts = []
        for p in perturbations:
            class_str = "n" if p['class_label'] == 0 else "s"
            feature_index = p['feature'].split('_')[-1]
            shift_val = str(p['sigma_shift']).replace('.', 'p').replace('-', 'm')
            pert_str_parts.append(f"pert_f{feature_index}{class_str}_by{shift_val}s")
        name_parts.append("_".join(pert_str_parts))


    if random_seed == TRAINING_SEED:
        name_parts.append("training")
    else:
        name_parts.append(f"seed{random_seed}")

    
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
        n_initial = ds_settings.get("n_initial_features", 0)
        total_features = n_initial

        # Quantitative Perturbation description
        pert_settings = config.get("perturbation_settings")
        if pert_settings and isinstance(pert_settings, list):
            pert_descs = []
            for p in pert_settings:
                feature = p.get('feature', 'N/A')
                class_label = 'Noise' if p.get('class_label') == 0 else 'Signal'
                sigma_shift = p.get('sigma_shift', 0.0)
                pert_descs.append(f"{feature} ({class_label}) by {sigma_shift:+.1f}σ")
            pert_desc = f"Perturbation: {'; '.join(pert_descs)}"
        else:
            pert_desc = "No Perturbations"

        # Separation calculation
        class_config = config.get("create_feature_based_signal_noise_classification", {})
        signal_features = class_config.get("signal_features", {})
        noise_features = class_config.get("noise_features", {})
        
        separations = [
            abs(signal_features[f]['mean'] - noise_features.get(f, {}).get('mean', 0))
            for f in signal_features if f in noise_features
        ]
        avg_separation = sum(separations) / len(separations) if separations else 0.0
        separation_desc = f"Avg. Separation: {avg_separation:.2f}"

        # Assemble the subtitle
        subtitle = (
            f"Dataset: {n_samples:,} Samples, {total_features} Features | "
            f"{pert_desc} | {separation_desc}"
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
