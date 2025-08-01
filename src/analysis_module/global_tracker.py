# src/analysis_module/global_tracker.py

import pandas as pd
import yaml
from pathlib import Path
import re
from src.data_generator_module.utils import find_project_root, load_yaml_config

def generate_global_tracking_sheet():
    """
    Scans for all completed experiment families, gathers their summary statistics
    and configuration details, and compiles them into a single master CSV file.
    This version correctly locates data configuration files.
    """
    project_root = Path(find_project_root())
    spreadsheets_dir = project_root / "reports" / "spreadsheets"
    generated_configs_dir = project_root / "configs" / "training" / "generated"

    if not spreadsheets_dir.exists():
        print(f"Error: Spreadsheets directory not found at '{spreadsheets_dir}'.")
        print("Please run the 'aggregate' or 'aggregate-all' command first.")
        return

    all_summary_files = list(spreadsheets_dir.glob("**/*_summary.csv"))
    if not all_summary_files:
        print(f"Error: No summary CSV files found in '{spreadsheets_dir}'.")
        return

    print(f"Found {len(all_summary_files)} experiment families to track.")
    
    tracking_data = []

    for summary_file in all_summary_files:
        # The stem is the family name, e.g., 'n1000..._pert...'
        family_name = summary_file.stem.replace('_summary', '')
        
        # 1. Load summary stats from the CSV
        summary_df = pd.read_csv(summary_file, index_col=0)
        
        # 2. --- MODIFICATION: Correctly locate the data generation config file ---
        # We now explicitly look for the seed0 config, as it's a reliable representative for the family.
        data_config_path = project_root / "configs" / "data_generation" / f"{family_name}_seed0_config.yml"
        
        if not data_config_path.exists():
            print(f"Warning: Could not find data config for family '{family_name}' at '{data_config_path}'. Skipping.")
            continue

        data_config = load_yaml_config(data_config_path)
        
        # 3. Dynamically find the corresponding optimal training config (from previous fix)
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
        tracking_data.append(row)

    # 5. Create and save the global DataFrame
    if not tracking_data:
        print("No data was collected for the global tracking sheet. Exiting.")
        return
        
    global_df = pd.DataFrame(tracking_data)
    
    leading_cols = [
        "experiment_family", "model", "n_samples", "n_features", "continuous", "discrete",
        "separation", "perturbation", "optimal_trial_num"
    ]
    metric_cols = sorted(list(set([col.replace('_mean', '') for col in global_df.columns if col.endswith('_mean')])))
    
    ordered_columns = leading_cols.copy()
    for metric in metric_cols:
        if f"{metric}_mean" in global_df.columns:
            ordered_columns.append(f"{metric}_mean")
        if f"{metric}_std" in global_df.columns:
            ordered_columns.append(f"{metric}_std")

    remaining_cols = [col for col in global_df.columns if col not in ordered_columns]
    final_ordered_columns = ordered_columns + remaining_cols
    
    final_ordered_columns = [col for col in final_ordered_columns if col in global_df.columns]
    global_df = global_df[final_ordered_columns]

    output_path = project_root / "reports" / "global_experiment_tracking.csv"
    global_df.to_csv(output_path, index=False)
    
    print("\n✓ Global experiment tracking sheet generated successfully!")
    print(f"Saved to: {output_path}")

