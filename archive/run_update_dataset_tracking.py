# run_update_dataset_tracking.py

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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    return round(os.path.getsize(file_path) / (1024 * 1024), 2)

def update_dataset_registry():
    """
    Intelligently updates the dataset tracking spreadsheet, syncing it with
    the contents of the data folder and populating all detail sheets.
    """
    project_root = Path(find_project_root())
    configs_dir = project_root / "configs" / "data_generation"
    data_dir = project_root / "data"
    output_file = project_root / "dataset_tracking.xlsx"
    sheet_names = [
        "Dataset Registry", "Configuration Details",
        "Feature Separation Details", "File Metadata"
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

    # --- 2. Audit and Remove Deleted Datasets ---
    registry_df = all_sheets["Dataset Registry"]
    if not registry_df.empty:
        registry_files = set(registry_df["Generated Filename"])
        physical_files = {f.name for f in data_dir.glob("*_dataset.csv")}
        files_to_remove = registry_files - physical_files
        
        if files_to_remove:
            print(f"\nAuditing registry... Found {len(files_to_remove)} dataset(s) to remove:")
            for f in files_to_remove:
                print(f" - Removing record for deleted file: {f}")
            ids_to_remove = set(registry_df[registry_df["Generated Filename"].isin(files_to_remove)]["Dataset ID"])
            all_sheets["Dataset Registry"] = registry_df[~registry_df["Generated Filename"].isin(files_to_remove)]
            for name in sheet_names[1:]:
                if "Dataset ID" in all_sheets[name].columns:
                    all_sheets[name] = all_sheets[name][~all_sheets[name]["Dataset ID"].isin(ids_to_remove)]
        
        existing_files_after_audit = set(all_sheets["Dataset Registry"]["Generated Filename"])
        print(f"\nFound {len(existing_files_after_audit)} datasets in registry after audit.")
    else:
        existing_files_after_audit = set()
        print("\nRegistry is empty. Scanning for all available datasets.")

    # --- 3. Scan for New Datasets to Add ---
    new_datasets_added = 0
    config_files = list(configs_dir.glob("*.yml")) + list(configs_dir.glob("*.yaml"))
    
    for config_file in config_files:
        try:
            config = load_config(config_file)
            if "create_feature_based_signal_noise_classification" not in config:
                continue
            
            expected_filename = f"{create_filename_from_config(config)}_dataset.csv"
            dataset_path = data_dir / expected_filename
            
            if dataset_path.exists() and expected_filename not in existing_files_after_audit:
                new_datasets_added += 1
                print(f" + Adding new dataset: {expected_filename}")

                dataset_id = f"DS{(len(existing_files_after_audit) + new_datasets_added):03d}"
                global_settings = config.get("global_settings", {})
                dataset_settings = config.get("dataset_settings", {})
                class_config = config["create_feature_based_signal_noise_classification"]

                separations = [abs(class_config["signal_features"][f]['mean'] - class_config["noise_features"].get(f, {}).get('mean', 0)) for f in class_config["signal_features"]]
                avg_sep = sum(separations) / len(separations) if separations else 0
                
                # Append to "Dataset Registry"
                registry_entry = pd.DataFrame([{"Dataset ID": dataset_id, "Config File": config_file.name, "Generated Filename": expected_filename, "Creation Date": datetime.fromtimestamp(dataset_path.stat().st_mtime).strftime("%Y-%m-%d"), "Description": f"{dataset_settings.get('n_samples', 0):,} samples, {dataset_settings.get('n_initial_features', 0)} features, Avg. Sep: {avg_sep:.2f}"}])
                all_sheets["Dataset Registry"] = pd.concat([all_sheets["Dataset Registry"], registry_entry], ignore_index=True)

                # Append to "Configuration Details"
                config_entry = pd.DataFrame([{"Dataset ID": dataset_id, "Random Seed": global_settings.get("random_seed"), "Num Samples": dataset_settings.get("n_samples"), "Num Initial Features": dataset_settings.get("n_initial_features")}])
                all_sheets["Configuration Details"] = pd.concat([all_sheets["Configuration Details"], config_entry], ignore_index=True)

                # Append to "Feature Separation Details"
                feature_details = []
                for feature, s_params in class_config["signal_features"].items():
                    n_params = class_config["noise_features"].get(feature, {})
                    separation = abs(s_params.get('mean', 0) - n_params.get('mean', 0))
                    feature_details.append({"Dataset ID": dataset_id, "Feature Name": feature, "Signal Mean": s_params.get('mean'), "Noise Mean": n_params.get('mean'), "Separation": separation, "Signal Std": s_params.get('std'), "Noise Std": n_params.get('std')})
                all_sheets["Feature Separation Details"] = pd.concat([all_sheets["Feature Separation Details"], pd.DataFrame(feature_details)], ignore_index=True)

                # Append to "File Metadata"
                metadata_entry = pd.DataFrame([{"Dataset ID": dataset_id, "File Size (MB)": get_file_size_mb(dataset_path), "MD5 Checksum": calculate_file_checksum(dataset_path), "Full Path": str(dataset_path)}])
                all_sheets["File Metadata"] = pd.concat([all_sheets["File Metadata"], metadata_entry], ignore_index=True)

        except Exception as e:
            print(f"Error processing {config_file.name}: {e}")

    # --- 4. Save All DataFrames Back to the Excel File ---
    if new_datasets_added > 0 or ('files_to_remove' in locals() and files_to_remove):
        # Ensure all DataFrames have the correct columns before saving
        for name, cols in [("Dataset Registry", ["Dataset ID", "Config File", "Generated Filename", "Creation Date", "Description"]),
                           ("Configuration Details", ["Dataset ID", "Random Seed", "Num Samples", "Num Initial Features"]),
                           ("Feature Separation Details", ["Dataset ID", "Feature Name", "Signal Mean", "Noise Mean", "Separation", "Signal Std", "Noise Std"]),
                           ("File Metadata", ["Dataset ID", "File Size (MB)", "MD5 Checksum", "Full Path"])]:
            if name in all_sheets:
                all_sheets[name] = all_sheets[name].reindex(columns=cols)

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for sheet_name, df in all_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\nDataset tracking spreadsheet updated: {output_file}")
    else:
        print("\nRegistry is already up-to-date.")

if __name__ == "__main__":
    update_dataset_registry()
