"""
Dataset tracking spreadsheet generator for the Gaussian data generator project.
"""

import os
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_generator_module.utils import find_project_root, create_filename_from_config


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_file_checksum(file_path):
    """Calculate MD5 checksum of file."""
    if not os.path.exists(file_path):
        return "N/A"

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:10]


def get_file_size_mb(file_path):
    """Get file size in MB."""
    if not os.path.exists(file_path):
        return 0
    return round(os.path.getsize(file_path) / (1024 * 1024), 1)


def create_dataset_registry():
    """Create comprehensive dataset tracking spreadsheet."""

    project_root = Path(find_project_root())
    configs_dir = project_root / "configs" / "data_generation"
    data_dir = project_root / "data"
    figures_dir = project_root / "reports" / "figures"

    # Initialize data structures
    main_registry = []
    config_details = []
    feature_details = []
    weight_stats = []
    file_metadata = []

    dataset_id = 1

    # Process each config file
    config_files = list(configs_dir.glob("*.yml")) + list(configs_dir.glob("*.yaml"))

    for config_file in config_files:
        try:
            config = load_config(config_file)

            # Generate expected filenames using your existing function
            expected_filename = create_filename_from_config(config)
            dataset_path = data_dir / f"{expected_filename}_dataset.csv"
            plot_path = figures_dir / f"{expected_filename}_plot.pdf"

            # Extract configuration details
            global_settings = config.get("global_settings", {})
            dataset_settings = config.get("dataset_settings", {})
            feature_generation = config.get("feature_generation", {})
            create_target = config.get("create_target", {})
            perturbation = config.get("perturbation", {})

            n_samples = dataset_settings.get("n_samples", 0)
            n_features = dataset_settings.get("n_initial_features", 0)
            random_seed = global_settings.get("random_seed", 42)

            feature_params = feature_generation.get("feature_parameters", {})
            feature_types = feature_generation.get("feature_types", {})

            weights = create_target.get("weights", [])
            function_type = create_target.get("function_type", "linear")
            noise_level = create_target.get("noise_level", 0.0)
            features_to_use = create_target.get(
                "features_to_use", list(feature_params.keys())
            )

            # Count feature types
            continuous_count = sum(
                1 for ft in feature_types.values() if ft == "continuous"
            )
            discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")

            # Perturbation info
            pert_enabled = bool(perturbation.get("features"))
            pert_type = perturbation.get("perturbation_type", "none")
            pert_scale = perturbation.get("scale", 0.0)

            # Weight statistics
            if weights:
                pos_count = sum(1 for w in weights if w > 0)
                neg_count = sum(1 for w in weights if w < 0)
                zero_count = sum(1 for w in weights if w == 0)
                min_weight = min(weights)
                max_weight = max(weights)
                weight_range = max_weight - min_weight
            else:
                pos_count = neg_count = zero_count = 0
                min_weight = max_weight = weight_range = 0

            # File metadata
            dataset_size = get_file_size_mb(dataset_path)
            checksum = calculate_file_checksum(dataset_path)

            # Status determination
            status = "Complete" if dataset_path.exists() else "Missing"

            # Main registry entry
            main_registry.append(
                {
                    "Dataset ID": f"DS{dataset_id:03d}",
                    "Config File": config_file.name,
                    "Generated Filename": f"{expected_filename}_dataset.csv",
                    "Creation Date": datetime.now().strftime("%Y-%m-%d"),
                    "Status": status,
                    "Description": f"{n_samples:,} samples, {n_features} features, {function_type} target",
                }
            )

            # Configuration details
            config_details.append(
                {
                    "Dataset ID": f"DS{dataset_id:03d}",
                    "Config File": config_file.name,
                    "Random Seed": random_seed,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "continuous_count": continuous_count,
                    "discrete_count": discrete_count,
                    "function_type": function_type,
                    "noise_level": noise_level,
                    "perturbations_enabled": pert_enabled,
                    "perturbation_type": pert_type,
                    "perturbation_scale": pert_scale,
                }
            )

            # Feature details
            for feature_name, params in feature_params.items():
                feature_weight = 0
                if feature_name in features_to_use:
                    idx = features_to_use.index(feature_name)
                    if idx < len(weights):
                        feature_weight = weights[idx]

                feature_details.append(
                    {
                        "Dataset ID": f"DS{dataset_id:03d}",
                        "Feature Name": feature_name,
                        "Feature Type": feature_types.get(feature_name, "unknown"),
                        "Mean": params.get("mean", 0),
                        "Std": params.get("std", 1),
                        "Weight": feature_weight,
                        "Used in Target": feature_name in features_to_use,
                    }
                )

            # Weight statistics
            weight_stats.append(
                {
                    "Dataset ID": f"DS{dataset_id:03d}",
                    "Total Weights": len(weights),
                    "Positive Count": pos_count,
                    "Negative Count": neg_count,
                    "Zero Count": zero_count,
                    "Min Weight": min_weight,
                    "Max Weight": max_weight,
                    "Weight Range": weight_range,
                }
            )

            # File metadata
            file_metadata.append(
                {
                    "Dataset ID": f"DS{dataset_id:03d}",
                    "Config Path": str(config_file.relative_to(project_root)),
                    "Dataset Path": str(dataset_path.relative_to(project_root)),
                    "Plot Path": str(plot_path.relative_to(project_root)),
                    "File Size (MB)": dataset_size,
                    "Checksum": checksum,
                    "Notes": f"{function_type.title()} function, {continuous_count}C/{discrete_count}D features",
                }
            )

            dataset_id += 1

        except Exception as e:
            print(f"Error processing {config_file}: {e}")
            continue

    # Create DataFrames and save to Excel
    output_file = project_root / "dataset_tracking.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        pd.DataFrame(main_registry).to_excel(
            writer, sheet_name="Dataset Registry", index=False
        )
        pd.DataFrame(config_details).to_excel(
            writer, sheet_name="Configuration Details", index=False
        )
        pd.DataFrame(feature_details).to_excel(
            writer, sheet_name="Feature Details", index=False
        )
        pd.DataFrame(weight_stats).to_excel(
            writer, sheet_name="Weight Statistics", index=False
        )
        pd.DataFrame(file_metadata).to_excel(
            writer, sheet_name="File Metadata", index=False
        )

    print(f"Dataset tracking spreadsheet created: {output_file}")
    print(f"Processed {len(main_registry)} datasets")


if __name__ == "__main__":
    create_dataset_registry()
