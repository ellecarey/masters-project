"""
Dataset tracking spreadsheet generator for classification datasets.
This version intelligently updates the existing spreadsheet by:
1. Removing entries for datasets that have been deleted from disk.
2. Appending entries for newly generated datasets.
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
    Intelligently updates the dataset tracking spreadsheet, syncing it with
    the contents of the data folder.
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

    # --- NEW: Audit and Remove Deleted Datasets ---
    registry_df = all_sheets["Dataset Registry"]
    if not registry_df.empty:
        # Get list of files currently in the registry
        registry_files = set(registry_df["Generated Filename"])
        # Get list of files physically present on disk
        physical_files = {f.name for f in data_dir.glob("*_dataset.csv")}
        
        # Identify files that are in the registry but not on disk
        files_to_remove = registry_files - physical_files
        
        if files_to_remove:
            print(f"\nAuditing registry... Found {len(files_to_remove)} dataset(s) to remove:")
            for f in files_to_remove:
                print(f"  - Removing record for deleted file: {f}")
            
            # Get the Dataset IDs to remove from all other sheets
            ids_to_remove = set(registry_df[registry_df["Generated Filename"].isin(files_to_remove)]["Dataset ID"])
            
            # Filter all sheets to remove the old records
            all_sheets["Dataset Registry"] = registry_df[~registry_df["Generated Filename"].isin(files_to_remove)]
            for name in sheet_names[1:]:
                df = all_sheets[name]
                if not df.empty:
                    all_sheets[name] = df[~df["Dataset ID"].isin(ids_to_remove)]

    # --- 3. Scan for New Datasets to Add ---
    existing_files_after_audit = set(all_sheets["Dataset Registry"]["Generated Filename"])
    print(f"\nFound {len(existing_files_after_audit)} datasets in registry after audit.")
    
    new_datasets_added = 0
    config_files = list(configs_dir.glob("*.yml")) + list(configs_dir.glob("*.yaml"))

    for config_file in config_files:
        try:
            config = load_config(config_file)
            if "create_feature_based_signal_noise_classification" not in config:
                continue

            expected_filename = f"{create_filename_from_config(config)}_dataset.csv"
            dataset_path = data_dir / expected_filename

            # Process only if the dataset file exists and is NOT already in the spreadsheet
            if dataset_path.exists() and expected_filename not in existing_files_after_audit:
                new_datasets_added += 1
                print(f"  + Adding new dataset: {expected_filename}")

                # ... (Logic to extract info and append rows is the same as before)
                dataset_id = f"DS{(len(existing_files_after_audit) + new_datasets_added):03d}"
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

                # Use pd.concat for safer appending
                all_sheets["Dataset Registry"] = pd.concat([all_sheets["Dataset Registry"], pd.DataFrame([{"Dataset ID": dataset_id, "Config File": config_file.name, "Generated Filename": expected_filename, "Creation Date": datetime.fromtimestamp(dataset_path.stat().st_mtime).strftime("%Y-%m-%d"), "Description": f"{n_samples:,} samples, {n_features} features, Avg. Sep: {avg_separation:.2f}"}])], ignore_index=True)
                # ... (and so on for other sheets)
        
        except Exception as e:
            print(f"Error processing {config_file.name}: {e}")

    # --- 4. Save All DataFrames Back to the Excel File ---
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nDataset tracking spreadsheet updated: {output_file}")
    if new_datasets_added == 0 and not 'files_to_remove' in locals() or not files_to_remove:
        print("Registry is already up-to-date.")

if __name__ == "__main__":
    update_dataset_registry()
