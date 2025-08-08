import os
import shutil
from pathlib import Path
import yaml
import random
import numpy as np
import re
from src.utils.plotting_helpers import generate_subtitle_from_config

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
    separations = []
    for f_name, s_params in signal_features.items():
        if f_name in noise_features:
            n_params = noise_features[f_name]
            mean_diff = abs(s_params.get('mean', 0) - n_params.get('mean', 0))
            s_std = s_params.get('std', 1)
            n_std = n_params.get('std', 1)
            # Prevent division by zero if stds are missing or zero
            if (s_std**2 + n_std**2) > 0:
                # Standardized separation for one feature
                d = mean_diff / ((s_std**2 + n_std**2)**0.5)
                separations.append(d)

    # Combine individual separations into one overall metric (root sum square)
    overall_separation = (sum(d**2 for d in separations))**0.5 if separations else 0.0
    random_seed = global_settings.get("random_seed", 42)

    name_parts = [
        f"n{n_samples}",
        f"f_init{n_features}",
        f"cont{continuous_count}",
        f"disc{discrete_count}",
        f"sep{str(round(overall_separation, 1)).replace('.', 'p')}"
    ]

    if perturbations:
        pert_str_parts = []
        for p in perturbations:
            class_str = "n" if p['class_label'] == 0 else "s"
            
            # Handle different perturbation types
            pert_type = p.get('type', 'individual')
            
            if pert_type == 'correlated':
                # Handle correlated perturbations
                features = p.get('features', [])
                if len(features) <= 2:
                    feature_str = ''.join([f.split('_')[-1] for f in features])
                else:
                    feature_str = f"{len(features)}f"
                
                if 'scale_factor' in p:
                    scale_val = str(p['scale_factor']).replace('.', 'p')
                    pert_str_parts.append(f"pert_corr{feature_str}{class_str}_scale{scale_val}")
                elif 'sigma_shift' in p:
                    shift_val = str(p['sigma_shift']).replace('.', 'p').replace('-', 'm')
                    pert_str_parts.append(f"pert_corr{feature_str}{class_str}_by{shift_val}s")
            else:
                # Handle individual perturbations (existing code)
                feature_index = p['feature'].split('_')[-1]
                if 'scale_factor' in p:
                    scale_val = str(p['scale_factor']).replace('.', 'p')
                    pert_str_parts.append(f"pert_f{feature_index}{class_str}_scale{scale_val}")
                elif 'sigma_shift' in p:
                    shift_val = str(p['sigma_shift']).replace('.', 'p').replace('-', 'm')
                    pert_str_parts.append(f"pert_f{feature_index}{class_str}_by{shift_val}s")
                elif 'additive_noise' in p:
                    noise_val = str(p['additive_noise']).replace('.', 'p')
                    pert_str_parts.append(f"pert_f{feature_index}{class_str}_noise{noise_val}")
                elif 'multiplicative_factor' in p:
                    mult_val = str(p['multiplicative_factor']).replace('.', 'p')
                    pert_str_parts.append(f"pert_f{feature_index}{class_str}_mult{mult_val}")

        name_parts.append("_".join(pert_str_parts))

    if random_seed == TRAINING_SEED:
        name_parts.append("training")
    else:
        name_parts.append(f"seed{random_seed}")

    return "_".join(name_parts)


def create_plot_title_from_config(config: dict) -> tuple[str, str]:
    """Generates a main title and subtitle using the new centralized helper."""
    main_title = "Distribution of Generated Features"
    subtitle = generate_subtitle_from_config(config)
    return main_title, subtitle

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
