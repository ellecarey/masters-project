import argparse
import json
import pandas as pd
from pathlib import Path
import sys
import openpyxl
import matplotlib.pyplot as plt
import shutil
import re
import numpy as np

from .comparison import compare_families
from src.utils.filenames import metrics_filename
from src.utils.report_paths import reports_root, extract_family_base, experiment_family_path
from src.data_generator_module import utils as data_utils
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.utils.plotting_helpers import bounded_yerr, calculate_adaptive_ylimits, add_single_series_labels, format_plot_title, apply_decimal_formatters

# Ensure the src directory is in the Python path for utils
try:
    from src.data_generator_module.utils import find_project_root
    from src.data_generator_module.plotting_style import apply_custom_plot_style
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_generator_module.utils import find_project_root
    from src.data_generator_module.plotting_style import apply_custom_plot_style

    
def find_experiment_family(models_dir: Path, base_experiment_name: str):
    """
    Finds all metric files belonging to a specific experiment family.
    """
    pattern = f"{base_experiment_name.rsplit('_seed', 1)[0]}*_metrics.json"
    metric_files = list(models_dir.glob(pattern))
    return metric_files
    
def extract_experiment_characteristics(experiment_name: str, data_config: dict) -> dict:
    """Extract key characteristics from experiment name and config for tracking."""
    
    # Extract from data config
    dataset_settings = data_config.get("dataset_settings", {})
    class_config = data_config.get("create_feature_based_signal_noise_classification", {})
    
    n_samples = dataset_settings.get("n_samples", 0)
    n_features = dataset_settings.get("n_initial_features", 0)
    
    # Count feature types
    feature_types = class_config.get("feature_types", {})
    continuous_count = sum(1 for ft in feature_types.values() if ft == "continuous")
    discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")
    
    # Calculate separation
    signal_features = class_config.get("signal_features", {})
    noise_features = class_config.get("noise_features", {})
    separations = [
        abs(signal_features[f]['mean'] - noise_features.get(f, {}).get('mean', 0))
        for f in signal_features if f in noise_features
    ]
    avg_separation = round(sum(separations) / len(separations), 2) if separations else 0.0
    
    # Determine if perturbed
    perturbation = "perturbed" if "_pert_" in experiment_name else "original"
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'continuous': continuous_count,
        'discrete': discrete_count,
        'separation': avg_separation,
        'perturbation': perturbation
    }

def aggregate(optimal_config: str):
    """
    Aggregates results from multi-seed training runs, saves them to a
    spreadsheet, and generates summary plots.
    """
    apply_custom_plot_style()
    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    
    # --- 1. Identify the Experiment Family ---
    optimal_config_path = Path(optimal_config)
    if not optimal_config_path.is_absolute():
        optimal_config_path = project_root / optimal_config_path
    if not optimal_config_path.exists():
        print(f"Error: Optimal config file not found at '{optimal_config_path}'")
        return

    base_experiment_name = optimal_config_path.stem.replace('_config', '')
    print(f"--- Aggregating results for experiment family: {base_experiment_name} ---")

    metric_files = find_experiment_family(models_dir, base_experiment_name)
    if not metric_files:
        print(f"Error: No metric files found for this experiment family in '{models_dir}'.")
        return
    print(f"Found {len(metric_files)} metric files to aggregate.")

    # --- 2. Load and Aggregate Metrics ---
    all_metrics = []
    # highlight-start
    for f in metric_files:
        # Use a regular expression to reliably find the seed number.
        match = re.search(r'_seed(\d+)', f.stem)
        
        # Only process files that have a valid seed number in their name.
        if match:
            try:
                seed = int(match.group(1))
                with open(f, 'r') as file:
                    metrics = json.load(file)
                metrics['seed'] = seed
                all_metrics.append(metrics)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping file '{f.name}' due to a processing error: {e}")
        else:
            # This will safely ignore files like the "..._training_..._metrics.json"
            print(f"Info: Skipping file '{f.name}' as it does not contain a seed number.")
    # highlight-end
    
    if not all_metrics:
        print("Error: No valid metric files could be processed.")
        return

    results_df = pd.DataFrame(all_metrics)
    # ... (The rest of the function for summarizing and plotting remains unchanged)
    results_df = results_df[['seed'] + [col for col in results_df.columns if col != 'seed']]
    summary_stats = results_df.drop(columns=['seed']).agg(['mean', 'std']).T
    summary_stats['mean'] = summary_stats['mean'].round(4)
    summary_stats['std'] = summary_stats['std'].round(4)

    print("\n--- Aggregated Results ---")
    print(results_df.round(4).to_string(index=False))
    print("\n--- Summary Statistics (Mean & Std Dev) ---")
    print(summary_stats)
    
    # ... (Saving to spreadsheet and plotting logic)
    try:
        base_config_name = base_experiment_name.replace('_mlp_001_optimal', '')
        data_config_path = project_root / "configs" / "data_generation" / f"{base_config_name}_config.yml"
        if data_config_path.exists():
            from src.data_generator_module import utils as data_utils
            data_config_dict = data_utils.load_yaml_config(data_config_path)
            experiment_characteristics = extract_experiment_characteristics(base_experiment_name, data_config_dict)
            experiment_row = {
                'experiment_family': base_experiment_name,
                'n_samples': experiment_characteristics['n_samples'],
                'n_features': experiment_characteristics['n_features'],
                'continuous': experiment_characteristics['continuous'],
                'discrete': experiment_characteristics['discrete'],
                'separation': experiment_characteristics['separation'],
                'perturbation': experiment_characteristics['perturbation'],
            }
            for metric in ['Test Loss (BCE)', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']:
                if metric in summary_stats.index:
                    experiment_row[f'{metric}_mean'] = summary_stats.loc[metric, 'mean']
                    experiment_row[f'{metric}_std'] = summary_stats.loc[metric, 'std']
            global_spreadsheet_path = reports_root() / "experiment_tracking_master.xlsx"
            global_spreadsheet_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if global_spreadsheet_path.exists():
                    existing_df = pd.read_excel(global_spreadsheet_path, sheet_name='Master_Tracking')
                    existing_df = existing_df[existing_df['experiment_family'] != base_experiment_name]
                    updated_df = pd.concat([existing_df, pd.DataFrame([experiment_row])], ignore_index=True)
                else:
                    updated_df = pd.DataFrame([experiment_row])
                updated_df = updated_df.sort_values('experiment_family').reset_index(drop=True)
                with pd.ExcelWriter(global_spreadsheet_path, engine='openpyxl') as writer:
                    updated_df.to_excel(writer, sheet_name='Master_Tracking', index=False)
                print(f"Global experiment tracking updated: {global_spreadsheet_path}")
            except Exception as e:
                print(f"Error updating global tracking spreadsheet: {e}")
        else:
            print(f"Warning: Could not find data config at {data_config_path}")
    except Exception as e:
        print(f"Error loading data config for experiment characteristics: {e}")

    # ... (Plotting logic)
    from src.utils.report_paths import extract_family_base, experiment_family_path
    family_base = extract_family_base(base_experiment_name)
    plot_data = summary_stats.reset_index().rename(columns={'index': 'Metric'})
    main_metrics_df = plot_data[plot_data['Metric'] != 'Test Loss (BCE)']
    yerr = bounded_yerr(main_metrics_df['mean'].values, main_metrics_df['std'].values)
    plt.errorbar(main_metrics_df['Metric'], main_metrics_df['mean'], yerr=yerr, fmt='-o', capsize=5, label='Mean ± Std Dev')
    value_labels = [f'{val:.3f}' for val in main_metrics_df['mean'].values]
    add_single_series_labels(x_positions=range(len(main_metrics_df)), values=main_metrics_df['mean'].values, labels_text=value_labels, color='black')
    title = format_plot_title("Aggregated Performance Metrics for", base_experiment_name)
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha="right")
    y_min, y_max = calculate_adaptive_ylimits(main_metrics_df['mean'].values, main_metrics_df['std'].values)
    plt.grid(True, linestyle='--', alpha=0.7)
    apply_decimal_formatters(plt.gca(), precision=3)
    plt.tight_layout()
    faceted_plot_path_main = experiment_family_path(
        full_experiment_name=base_experiment_name,
        art_type="figure",
        subfolder="aggregation_summary",
        filename=f"{base_experiment_name}_summary.pdf"
    )
    plt.savefig(faceted_plot_path_main)
    print(f"\nSaved main metrics plot to: {faceted_plot_path_main}")
    plt.close()

    print("\n--- Aggregation complete. ---")


def aggregate_all_families(optimal_config: str):
    """
    Aggregate both the original and all detected perturbed experiment families
    sharing the same base as the given optimal config, fully automatically.
    """
    # --- Setup ---
    orig_opt_path = Path(optimal_config)
    if not orig_opt_path.exists():
        print(f"Error: The provided optimal config file does not exist: {orig_opt_path}")
        return

    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    
    # --- 1. Find base family info from the provided config name ---
    orig_base_name = orig_opt_path.stem
    base_prefix = re.sub(r'_(?:seed\d+|training)_mlp_.*_optimal$', '', orig_base_name)
    print(f"Scanning for experiment families with base: {base_prefix}")

    # --- 2. Collect all relevant metric files to discover families ---
    all_metric_files = list(models_dir.glob(f"{base_prefix}*_metrics.json"))
    if not all_metric_files:
        print("No matching metric files found.")
        return

    family_bases = {
        mf.stem.split("_seed")[0] for mf in all_metric_files if "_seed" in mf.stem
    }

    # --- 3. Extract model suffix from the provided optimal config name ---
    model_suffix_match = re.search(r'(_mlp_.*_optimal)$', orig_opt_path.stem)
    if not model_suffix_match:
        print(f"Error: Could not determine model suffix from '{orig_opt_path.name}'")
        return
    model_suffix = model_suffix_match.group(1)

    # --- 4. Loop through discovered families and aggregate ---
    for family in sorted(family_bases):
        family_config_filename = f"{family}_seed0{model_suffix}.yml"
        family_config_path = orig_opt_path.parent / family_config_filename

        if not family_config_path.exists():
            print(f"Creating temporary config '{family_config_path.name}' for aggregation.")
            shutil.copyfile(orig_opt_path, family_config_path)

        print(f"\nAggregating family: {family}")
        aggregate(str(family_config_path))

    # --- Suggest the comparison step ---
    print("\n--- Auto-aggregation of all original and perturbed experiment families complete. ---")

    perturbation_tag = None
    for family in family_bases:
        if "_pert_" in family:
            match = re.search(r'_(pert_.*)$', family)
            if match:
                perturbation_tag = match.group(1)
                break
    
    if perturbation_tag:
        print("\n" + "="*80)
        print("Next Step: Compare the aggregated results")
        print("="*80 + "\n")
        print("Use the 'compare-families' command to generate plots comparing the performance")
        print("of the original dataset against the perturbed version.")
        
        # We need to construct the path to the original family's optimal config for the next command.
        # The user provided the training config, but compare-families needs one from the family.
        original_family_optimal_config = orig_opt_path.parent / f"{base_prefix}_seed0{model_suffix}.yml"
        
        compare_command = (
            f"uv run experiment_manager.py compare-families \\\n"
            f"    --original-optimal-config {original_family_optimal_config} \\\n"
            f"    --perturbation-tag {perturbation_tag}"
        )
        print(compare_command)
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(
        description="CLI tools for experiment analysis (aggregation and comparison)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Aggregate single family (original OR one perturbed)
    agg = subparsers.add_parser(
        "aggregate", help="Aggregate multi-seed results for an experiment family."
    )
    agg.add_argument(
        "--optimal-config", required=True, type=str,
        help="Path to one of the final optimal config files used for the runs (e.g., ..._optimal.yml)."
    )

    # Compare-families
    comp = subparsers.add_parser(
        "compare-families", help="Compare original and perturbed experiment families."
    )
    comp.add_argument(
        "--original-optimal-config", required=True, type=str,
        help="Original (unperturbed) optimal config YAML file."
    )
    comp.add_argument(
        "--perturbation-tag", required=True, type=str,
        help="Perturbation tag used in filenames for the perturbed family (e.g. 'pert_f4n_by1p0s')."
    )

    # Fully auto aggregation (all families discovered by pattern)
    agg_all = subparsers.add_parser(
        "aggregate-all",
        help="Aggregate original and all detected perturbed experiment families automatically."
    )
    agg_all.add_argument(
        "--optimal-config", required=True, type=str,
        help="Path to the optimal config file of the ORIGINAL family."
    )

    args = parser.parse_args()

    if args.command == "aggregate":
        aggregate(args.optimal_config)
    elif args.command == "compare-families":
        compare_families(args.original_optimal_config, args.perturbation_tag)
    elif args.command == "aggregate-all":
        aggregate_all_families(args.optimal_config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()