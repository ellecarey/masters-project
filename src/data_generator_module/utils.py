import os
from pathlib import Path
import yaml


def find_project_root():
    """Find the project root by searching upwards for a marker file."""
    # Start from the directory of this file (__file__).
    current_path = Path(__file__).resolve()

    # Define project root markers. These files exist in your project's root.
    markers = [".git", "pyproject.toml", "README.md", "run_generator.py"]

    for parent in current_path.parents:
        # Check if any marker file exists in the current parent directory.
        if any((parent / marker).exists() for marker in markers):
            # If a marker is found, we have found the project root.
            print(f"Project root found at: {parent}")
            return str(parent)

    # --- FALLBACK ---
    # last resort if no markers are found
    # assumes a fixed structure: utils.py -> generator_package -> src -> masters-project
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


# This function might also be in your utils.py and it should stay the same
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
    """
    Generates a descriptive filename prefix from the configuration dictionary.
    """
    try:
        # Extract key parameters from the config
        ds_settings = config.get("dataset_settings", {})
        n_samples = ds_settings.get("n_samples", "N/A")
        n_initial = ds_settings.get("n_initial_features", "N/A")

        add_features_settings = config.get("add_features", {})
        n_added = add_features_settings.get("n_new_features", 0)

        pert_settings = config.get("perturbation", {})
        pert_type = pert_settings.get("perturbation_type", "none")
        pert_scale = pert_settings.get("scale", 0)

        target_settings = config.get("create_target", {})
        func_type = target_settings.get("function_type", "linear")

        # Sanitise perturbation scale for filename
        scale_str = str(pert_scale).replace(".", "p")

        # Construct the filename
        # e.g., n1000_f_init5_add2_pert-gaussian_scl0p15_func-polynomial
        filename = (
            f"n{n_samples}_"
            f"f_init{n_initial}_add{n_added}_"
            f"pert-{pert_type}_scl{scale_str}_"
            f"func-{func_type}"
        )
        return filename
    except Exception:
        # Fallback to a generic name if config structure is unexpected
        return "manually_name_dataset"
