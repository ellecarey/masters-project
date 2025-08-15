import pandas as pd
import yaml
from pathlib import Path
import re
from src.data_generator_module.utils import find_project_root, load_yaml_config
import numpy as np
import copy

def _calculate_separation_from_config(config: dict) -> float:
    """
    Calculates the overall dataset separation from a data config dict.
    """
    class_config = config.get("create_feature_based_signal_noise_classification", {})
    signal_features = class_config.get("signal_features", {})
    noise_features = class_config.get("noise_features", {})
    
    separations = []
    for f_name, s_params in signal_features.items():
        if f_name in noise_features:
            n_params = noise_features[f_name]
            mean_diff = abs(s_params.get('mean', 0) - n_params.get('mean', 0))
            s_std = s_params.get('std', 1)
            n_std = n_params.get('std', 1)
            if (s_std**2 + n_std**2) > 0:
                d = mean_diff / ((s_std**2 + n_std**2)**0.5)
                separations.append(d)
                
    overall_separation = (sum(d**2 for d in separations))**0.5 if separations else 0.0
    return overall_separation

def _calculate_perturbed_separation_in_memory(original_config: dict, perturbation_settings: list) -> float:
    """
    Calculates the dataset separation by applying perturbations to the original
    feature parameters in memory, without modifying any files.
    """
    if not perturbation_settings:
        return _calculate_separation_from_config(original_config)

    # Deep copy the feature parameters to avoid side effects
    temp_signal_features = copy.deepcopy(original_config['create_feature_based_signal_noise_classification']['signal_features'])
    temp_noise_features = copy.deepcopy(original_config['create_feature_based_signal_noise_classification']['noise_features'])

    # Simulate the effect of the perturbations on the parameters
    for p_conf in perturbation_settings:
        feature_name = p_conf.get('feature')
        class_label = p_conf.get('class_label')
        sigma_shift = p_conf.get('sigma_shift')
        scale_factor = p_conf.get('scale_factor')

        if not feature_name:
            continue

        target_features = temp_signal_features if class_label == 1 else temp_noise_features
        if feature_name in target_features:
            params_to_update = target_features[feature_name]
            if sigma_shift is not None:
                std_dev = params_to_update.get('std', 1.0)
                params_to_update['mean'] += sigma_shift * std_dev
            if scale_factor is not None:
                params_to_update['mean'] *= scale_factor
                params_to_update['std'] *= scale_factor
    
    temp_config_for_calc = {'create_feature_based_signal_noise_classification': {'signal_features': temp_signal_features, 'noise_features': temp_noise_features}}
    return _calculate_separation_from_config(temp_config_for_calc)

    
def generate_global_tracking_sheet():
    """
    Scans for all completed experiment families, gathers their summary statistics
    and configuration details, and compiles them into a single master CSV file.
    """
    project_root = Path(find_project_root())
    spreadsheets_dir = project_root / "reports" / "spreadsheets"
    generated_configs_dir = project_root / "configs" / "training" / "generated"
    data_gen_configs_dir = project_root / "configs" / "data_generation"
    output_path = project_root / "reports" / "global_experiment_tracking.csv"

    if not spreadsheets_dir.exists():
        print(f"Error: Spreadsheets directory not found at '{spreadsheets_dir}'.")
        return

    all_summary_files = list(spreadsheets_dir.glob("**/*_summary.csv"))
    if not all_summary_files:
        print(f"Info: No new summary CSV files found in '{spreadsheets_dir}'. Global tracker will not be modified.")
        return

    print(f"Found {len(all_summary_files)} experiment families to process for the tracking sheet.")
    new_tracking_data = []

    for summary_file in all_summary_files:
        family_name = summary_file.stem.replace('_summary', '')
        summary_df = pd.read_csv(summary_file, index_col=0)

        data_config_path = data_gen_configs_dir / f"{family_name}_seed0_config.yml"
        if not data_config_path.exists():
            print(f"Warning: Could not find data config for family '{family_name}' at '{data_config_path}'. Skipping.")
            continue
        data_config = load_yaml_config(data_config_path)

        # --- Final Corrected Logic for Separation Calculation ---
        separation_for_display = np.nan
        separation_delta = np.nan

        if "_pert_" in family_name:
            # For perturbed families, the 'separation' column must show the ORIGINAL separation.
            original_family_base = family_name.split('_pert_')[0]
            original_config_path = data_gen_configs_dir / f"{original_family_base}_seed0_config.yml"
            
            if original_config_path.exists():
                original_data_config = load_yaml_config(original_config_path)
                
                # This is the value for the 'separation' column. It comes from the original config.
                separation_for_display = _calculate_separation_from_config(original_data_config)
                
                # Now, calculate the perturbed separation in memory to compute the delta.
                perturbation_settings = data_config.get("perturbation_settings", [])
                perturbed_separation = _calculate_perturbed_separation_in_memory(original_data_config, perturbation_settings)
                
                # The delta is the difference between the new and original separations.
                separation_delta = perturbed_separation - separation_for_display
            else:
                print(f"Warning: Could not find original config '{original_config_path.name}' to calculate separation delta.")
                # Fallback: display the separation from the current (perturbed) file's params.
                separation_for_display = _calculate_separation_from_config(data_config)
        else:
            # For original, unperturbed families, 'separation' is just its own value.
            separation_for_display = _calculate_separation_from_config(data_config)
        
        # --- Logic to find optimal config, model, etc. (This is correct) ---
        base_family_name = re.sub(r'(_pert_.*|_seed.*)', '', family_name)
        optimal_config_pattern = f"{base_family_name}_*_optimal.yml"
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
            model_name = optimal_config.get("training_settings", {}).get("model_name", "N/A")
            optimal_trial_number = optimal_config.get("training_settings", {}).get("optimal_trial_number", "N/A")
        
        # --- Extract other required details ---
        dataset_settings = data_config.get("dataset_settings", {})
        class_config = data_config.get("create_feature_based_signal_noise_classification", {})
        n_samples = dataset_settings.get("n_samples", "N/A")
        n_features = dataset_settings.get("n_initial_features", "N/A")
        feature_types = class_config.get("feature_types", {})
        continuous = sum(1 for v in feature_types.values() if v == 'continuous')
        discrete = sum(1 for v in feature_types.values() if v == 'discrete')
        
        # --- Perturbation description logic ---
        perturbation_info = "original"
        if "perturbation_settings" in data_config and data_config["perturbation_settings"]:
            pert_descs = []
            for p_config in data_config["perturbation_settings"]:
                p_type = p_config.get("type", "individual")
                class_label_raw = p_config.get('class_label')
                class_label = 'noise' if class_label_raw == 0 else 'signal' if class_label_raw == 1 else 'N/A'
                if p_type == 'correlated':
                    features = p_config.get('features', [])
                    feature_str = '+'.join([f.replace('feature_', 'F') for f in features]) if len(features) <= 2 else f"{len(features)} feats"
                    desc = f"Corr {feature_str} ({class_label})"
                    if 'scale_factor' in p_config:
                        desc += f" by {p_config['scale_factor']}x"
                    elif 'sigma_shift' in p_config:
                        desc += f" by {p_config['sigma_shift']}s"
                    pert_descs.append(desc)
                else:
                    feature = p_config.get('feature', 'N/A').replace('feature_', 'F')
                    desc = f"{feature} ({class_label})"
                    if 'scale_factor' in p_config:
                        desc += f" scaled by {p_config['scale_factor']}x"
                    elif 'sigma_shift' in p_config:
                        desc += f" shifted by {p_config['sigma_shift']}s"
                    pert_descs.append(desc)
            perturbation_info = "; ".join(pert_descs)
            
        metrics_mean = summary_df['mean'].to_dict()
        metrics_std = summary_df['std'].to_dict()

        row = {
            "experiment_family": family_name,
            "model": model_name,
            "n_samples": n_samples,
            "n_features": n_features,
            "continuous": continuous,
            "discrete": discrete,
            "separation": f"{separation_for_display:.2f}",
            "separation_delta": f"{separation_delta:+.2f}" if not pd.isna(separation_delta) else "N/A",
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
    
    # --- Load existing data, then update and append new data ---
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, new_data_df]).drop_duplicates(subset=['experiment_family'], keep='last')
            global_df = combined_df.sort_values(by="experiment_family").reset_index(drop=True)
            print(f"Updated global tracking sheet. Total records: {len(global_df)}")
        except Exception as e:
            print(f"Warning: Could not merge with existing tracking sheet. A new one will be created from current data. Error: {e}")
            global_df = new_data_df
    else:
        print("Creating a new global tracking sheet.")
        global_df = new_data_df
        
    # --- Organize columns and save the final DataFrame ---
    leading_cols = [
        "experiment_family", "model", "n_samples", "n_features", "continuous", "discrete",
        "separation", "separation_delta", "perturbation", "optimal_trial_num"
    ]
    
    metric_cols = sorted(list(set([col.replace('_mean', '').replace('_std', '') for col in global_df.columns if col.endswith(('_mean', '_std')) and col not in leading_cols])))
    
    ordered_columns = leading_cols.copy()
    for metric in metric_cols:
        if f"{metric}_mean" in global_df.columns:
            ordered_columns.append(f"{metric}_mean")
        if f"{metric}_std" in global_df.columns:
            ordered_columns.append(f"{metric}_std")
            
    existing_cols = set(global_df.columns)
    final_ordered_columns = [col for col in ordered_columns if col in existing_cols]
    remaining_cols = sorted([col for col in existing_cols if col not in final_ordered_columns])
    final_ordered_columns.extend(remaining_cols)
    
    global_df = global_df[final_ordered_columns]
    
    global_df.to_csv(output_path, index=False)
    print("\n✓ Global experiment tracking sheet updated successfully!")
    print(f"Saved to: {output_path}")


