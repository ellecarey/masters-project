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
    # Extract dataset settings
    dataset_settings = config.get("dataset_settings", {})
    n_samples = dataset_settings.get("n_samples", 1000)
    n_initial_features = dataset_settings.get("n_initial_features", 5)

    # Extract feature type distribution from feature_generation
    feature_generation = config.get("feature_generation", {})
    feature_types = feature_generation.get("feature_types", {})

    # Count feature types
    continuous_count = sum(
        1 for ftype in feature_types.values() if ftype == "continuous"
    )
    discrete_count = sum(1 for ftype in feature_types.values() if ftype == "discrete")

    # Extract additional features
    add_features = config.get("add_features", {})
    n_new_features = add_features.get("n_new_features", 0)

    # Count additional feature types if they exist
    add_continuous = 0
    add_discrete = 0
    if n_new_features > 0:
        add_feature_types = add_features.get("feature_types", {})
        add_continuous = sum(
            1 for ftype in add_feature_types.values() if ftype == "continuous"
        )
        add_discrete = sum(
            1 for ftype in add_feature_types.values() if ftype == "discrete"
        )

    # Total feature type counts
    total_continuous = continuous_count + add_continuous
    total_discrete = discrete_count + add_discrete

    # Extract perturbation settings
    perturbation = config.get("perturbation", {})
    perturbation_type = perturbation.get("perturbation_type", "none")
    perturbation_scale = perturbation.get("scale", 0)

    # Extract target settings
    target = config.get("create_target", {})
    function_type = target.get("function_type", "linear")
    noise_level = target.get("noise_level", 0.1)

    # Create filename components with feature type distribution
    filename_parts = [
        f"n{n_samples}",
        f"f_init{n_initial_features}",
        f"cont{total_continuous}",
        f"disc{total_discrete}",
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

        # Target variable description
        func_type = config.get("create_target", {}).get("function_type", "N/A")
        target_desc = f"Target: {func_type.capitalize()} Relationship"

        # Assemble the subtitle
        subtitle = (
            f"Dataset: {n_samples} Samples, {total_features} Features | "
            f"{pert_desc} | {target_desc}"
        )

        return main_title, subtitle

    except Exception:
        # Fallback if the config structure is unexpected
        return "Feature Distribution", "Configuration details unavailable"
