import pandas as pd
import yaml
from pathlib import Path
import re
from src.data_generator_module.utils import find_project_root, load_yaml_config

def generate_global_tracking_sheet():
    """
    Scans for all completed experiment families, gathers their summary statistics
    and configuration details, and compiles them into a single master CSV file.
    This version loads the existing tracking sheet and updates it to preserve history.
    """
    project_root = Path(find_project_root())
    spreadsheets_dir = project_root / "reports" / "spreadsheets"
    generated_configs_dir = project_root / "configs" / "training" / "generated"
    output_path = project_root / "reports" / "global_experiment_tracking.csv"

    if not spreadsheets_dir.exists():
        print(f"Error: Spreadsheets directory not found at '{spreadsheets_dir}'.")
        print("Please run the 'aggregate' or 'aggregate-all' command first.")
        return

    all_summary_files = list(spreadsheets_dir.glob("**/*_summary.csv"))

    if not all_summary_files:
        print(f"Info: No new summary CSV files found in '{spreadsheets_dir}'. Global tracker will not be modified.")
        return

    print(f"Found {len(all_summary_files)} experiment families to process for the tracking sheet.")

    new_tracking_data = []
    for summary_file in all_summary_files:
        # The stem is the family name, e.g., 'n1000..._pert...'
        family_name = summary_file.stem.replace('_summary', '')

        # 1. Load summary stats from the CSV
        summary_df = pd.read_csv(summary_file, index_col=0)

        # 2. Correctly locate the data generation config file
        data_config_path = project_root / "configs" / "data_generation" / f"{family_name}_seed0_config.yml"
        if not data_config_path.exists():
            print(f"Warning: Could not find data config for family '{family_name}' at '{data_config_path}'. Skipping.")
            continue
        data_config = load_yaml_config(data_config_path)

        # 3. Dynamically find the corresponding optimal training config
        base_family_name = re.sub(r'(_pert_.*|_seed.*|_training.*)', '', family_name)
        optimal_config_pattern = f"{base_family_name}_training_*_optimal.yml"
        found_configs = list(generated_configs_dir.glob(optimal_config_pattern))

        optimal_trial_number = 'N/A'
        model_name = 'N/A'
        if not found_configs:
            print(f"Warning: Could not find an optimal config file for family base '{base_family_name}'.")
        else:
            if len(found_configs) > 1:
                print(f"Warning: Found multiple optimal configs for '{base_family_name}'. Using the first one: {found_configs[0]}")
            optimal_config_path = found_configs[0]
            optimal_config = load_yaml_config(optimal_config_path)
            optimal_trial_number = optimal_config.get("training_settings", {}).get("optimal_trial_number", "N/A")
            model_name = optimal_config.get("training_settings", {}).get("model_name", "N/A")

        # 4. Extract all required details for the spreadsheet row
        dataset_settings = data_config.get("dataset_settings", {})
        class_config = data_config.get("create_feature_based_signal_noise_classification", {})
        n_samples = dataset_settings.get("n_samples", "N/A")
        n_features = dataset_settings.get("n_initial_features", "N/A")
        feature_types = class_config.get("feature_types", {})
        continuous = sum(1 for v in feature_types.values() if v == 'continuous')
        discrete = sum(1 for v in feature_types.values() if v == 'discrete')
        signal_features = class_config.get("signal_features", {})
        noise_features = class_config.get("noise_features", {})
        separations = [abs(signal_features[f]['mean'] - noise_features.get(f, {}).get('mean', 0)) for f in signal_features if f in noise_features]
        avg_separation = sum(separations) / len(separations) if separations else 0.0

        perturbation_info = "original"
        if "perturbation_settings" in data_config:
            pert = data_config["perturbation_settings"][0]
            pert_class = 'noise' if pert.get('class_label') == 0 else 'signal'
            perturbation_info = f"perturbed: {pert.get('feature')} ({pert_class}) by {pert.get('sigma_shift')}s"

        metrics_mean = summary_df['mean'].to_dict()
        metrics_std = summary_df['std'].to_dict()

        row = {
            "experiment_family": family_name,
            "model": model_name,
            "n_samples": n_samples,
            "n_features": n_features,
            "continuous": continuous,
            "discrete": discrete,
            "separation": f"{avg_separation:.1f}",
            "perturbation": perturbation_info,
            "optimal_trial_num": optimal_trial_number,
            **{f"{k}_mean": v for k, v in metrics_mean.items()},
            **{f"{k}_std": v for k, v in metrics_std.items()}
        }
        new_tracking_data.append(row)

    if not new_tracking_data:
        print("No valid data was collected to update the global tracking sheet. Exiting.")
        return

    new_data_df = pd.DataFrame(new_tracking_data)

    # 5. Load existing data, then update and append new data
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            # Combine old and new data, replacing old rows with new ones if the experiment family matches
            combined_df = pd.concat([existing_df, new_data_df]).drop_duplicates(subset=['experiment_family'], keep='last')
            # Sort to keep the file tidy
            global_df = combined_df.sort_values(by="experiment_family").reset_index(drop=True)
            print(f"Updated global tracking sheet. Total records: {len(global_df)}")
        except Exception as e:
            print(f"Warning: Could not merge with existing tracking sheet. A new one will be created from current data. Error: {e}")
            global_df = new_data_df
    else:
        print("Creating a new global tracking sheet.")
        global_df = new_data_df

    # 6. Organize columns and save the final DataFrame
    leading_cols = [
        "experiment_family", "model", "n_samples", "n_features", "continuous", "discrete",
        "separation", "perturbation", "optimal_trial_num"
    ]
    
    metric_cols = sorted(list(set([col.replace('_mean', '').replace('_std', '') for col in global_df.columns if col.endswith(('_mean', '_std')) and col not in leading_cols])))
    
    ordered_columns = leading_cols.copy()
    for metric in metric_cols:
        if f"{metric}_mean" in global_df.columns:
            ordered_columns.append(f"{metric}_mean")
        if f"{metric}_std" in global_df.columns:
            ordered_columns.append(f"{metric}_std")

    # Ensure all columns from the DataFrame are included in the final list
    existing_cols = set(global_df.columns)
    final_ordered_columns = [col for col in ordered_columns if col in existing_cols]
    remaining_cols = sorted([col for col in existing_cols if col not in final_ordered_columns])
    final_ordered_columns.extend(remaining_cols)
    
    global_df = global_df[final_ordered_columns]

    global_df.to_csv(output_path, index=False)
    print("\n✓ Global experiment tracking sheet updated successfully!")
    print(f"Saved to: {output_path}")

