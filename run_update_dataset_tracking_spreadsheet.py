"""
Dataset tracking spreadsheet generator for classification datasets.
This version intelligently updates the existing spreadsheet, preserving
manual changes and appending only new datasets.
"""

import os
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import sys

# Ensure the src directory is in the Python path
try:
    from src.data_generator_module.utils import find_project_root, create_filename_from_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_generator_module.utils import find_project_root, create_filename_from_config


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_file_checksum(file_path):
    """Calculate MD5 checksum of a file."""
    if not os.path.exists(file_path):
        return "N/A"
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:10]

def get_file_size_mb(file_path):
    """Get file size in megabytes."""
    if not os.path.exists(file_path):
        return 0
    return round(os.path.getsize(file_path) / (1024 * 1024), 1)

def update_dataset_registry():
    """
    Intelligently updates the dataset tracking spreadsheet, adding new datasets
    and refreshing metadata for existing ones without overwriting manual data.
    """
    project_root = Path(find_project_root())
    configs_dir = project_root / "configs" / "data_generation"
    data_dir = project_root / "data"
    output_file = project_root / "dataset_tracking.xlsx"

    sheet_names = [
        "Dataset Registry",
        "Configuration Details",
        "Feature Separation Details",
        "File Metadata",
    ]

    # --- 1. Load Existing Spreadsheet or Create New DataFrames ---
    if output_file.exists():
        print(f"Loading existing spreadsheet: {output_file}")
        try:
            all_sheets = pd.read_excel(output_file, sheet_name=sheet_names)
        except ValueError:
            print("Warning: One or more sheets missing. Re-creating spreadsheet structure.")
            all_sheets = {name: pd.DataFrame() for name in sheet_names}
    else:
        print("No existing spreadsheet found. Creating a new one.")
        all_sheets = {name: pd.DataFrame() for name in sheet_names}

    existing_files = set(all_sheets["Dataset Registry"]["Generated Filename"]) if "Generated Filename" in all_sheets["Dataset Registry"] else set()
    print(f"Found {len(existing_files)} datasets already in the registry.")

    # --- 2. Scan for All Existing Datasets and Their Configs ---
    new_datasets_added = 0
    config_files = list(configs_dir.glob("*.yml")) + list(configs_dir.glob("*.yaml"))

    for config_file in config_files:
        try:
            config = load_config(config_file)
            if "create_feature_based_signal_noise_classification" not in config:
                continue

            expected_filename = f"{create_filename_from_config(config)}_dataset.csv"
            dataset_path = data_dir / expected_filename

            if dataset_path.exists() and expected_filename not in existing_files:
                new_datasets_added += 1
                print(f"  + Adding new dataset: {expected_filename}")

                dataset_id = f"DS{(len(existing_files) + new_datasets_added):03d}"
                global_settings = config.get("global_settings", {})
                dataset_settings = config.get("dataset_settings", {})
                class_config = config["create_feature_based_signal_noise_classification"]
                n_samples = dataset_settings.get("n_samples", 0)
                n_features = dataset_settings.get("n_initial_features", 0)
                random_seed = global_settings.get("random_seed", 42)
                feature_types = class_config.get("feature_types", {})
                signal_features = class_config.get("signal_features", {})
                noise_features = class_config.get("noise_features", {})
                
                separations = [abs(signal_features[f]['mean'] - noise_features.get(f, {}).get('mean', 0)) for f in signal_features if f in noise_features]
                avg_separation = sum(separations) / len(separations) if separations else 0.0

                all_sheets["Dataset Registry"] = pd.concat([all_sheets["Dataset Registry"], pd.DataFrame([{
                    "Dataset ID": dataset_id,
                    "Config File": config_file.name,
                    "Generated Filename": expected_filename,
                    "Creation Date": datetime.fromtimestamp(dataset_path.stat().st_mtime).strftime("%Y-%m-%d"),
                    "Description": f"{n_samples:,} samples, {n_features} features, Avg. Sep: {avg_separation:.2f}",
                }])], ignore_index=True)
                
                continuous_count = sum(1 for ft in feature_types.values() if ft == "continuous")
                discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")
                all_sheets["Configuration Details"] = pd.concat([all_sheets["Configuration Details"], pd.DataFrame([{
                    "Dataset ID": dataset_id, "Config File": config_file.name, "Random Seed": random_seed,
                    "n_samples": n_samples, "n_features": n_features,
                    "continuous_features": continuous_count, "discrete_features": discrete_count,
                    "avg_separation": f"{avg_separation:.2f}",
                }])], ignore_index=True)

                new_feature_details = []
                for feature_name, s_params in signal_features.items():
                    n_params = noise_features.get(feature_name, {})
                    new_feature_details.append({
                        "Dataset ID": dataset_id, "Feature Name": feature_name,
                        "Feature Type": feature_types.get(feature_name, "N/A"),
                        "Signal Mean": s_params.get('mean', 0), "Signal Std": s_params.get('std', 0),
                        "Noise Mean": n_params.get('mean', 0), "Noise Std": n_params.get('std', 0),
                        "Mean Separation": abs(s_params.get('mean', 0) - n_params.get('mean', 0)),
                    })
                all_sheets["Feature Separation Details"] = pd.concat([all_sheets["Feature Separation Details"], pd.DataFrame(new_feature_details)], ignore_index=True)
                
                all_sheets["File Metadata"] = pd.concat([all_sheets["File Metadata"], pd.DataFrame([{
                    "Dataset ID": dataset_id,
                    "Config Path": str(config_file.relative_to(project_root)),
                    "Dataset Path": str(dataset_path.relative_to(project_root)),
                    "File Size (MB)": get_file_size_mb(dataset_path),
                    "Checksum (MD5)": calculate_file_checksum(dataset_path),
                }])], ignore_index=True)

        except Exception as e:
            print(f"Error processing {config_file.name}: {e}")
            continue

    # --- 3. Save All DataFrames Back to the Excel File ---
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nDataset tracking spreadsheet updated: {output_file}")
    if new_datasets_added > 0:
        print(f"Successfully added {new_datasets_added} new dataset(s) to the registry.")
    else:
        print("No new datasets found to add.")


if __name__ == "__main__":
    update_dataset_registry()
