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
    for f in metric_files:
        with open(f, 'r') as file:
            metrics = json.load(file)
            seed = f.stem.split('_seed')[-1].split('_')[0]
            metrics['seed'] = int(seed)
            all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)
    results_df = results_df[['seed'] + [col for col in results_df.columns if col != 'seed']]
    
    # --- 3. Calculate Summary Statistics ---
    summary_stats = results_df.drop(columns=['seed']).agg(['mean', 'std']).T
    summary_stats['mean'] = summary_stats['mean'].round(4)
    summary_stats['std'] = summary_stats['std'].round(4)
    
    print("\n--- Aggregated Results ---")
    print(results_df.round(4).to_string(index=False))
    print("\n--- Summary Statistics (Mean & Std Dev) ---")
    print(summary_stats)
    
    # --- 4. Save to Global Experiment Tracking Spreadsheet ---
    try:
        # Load the corresponding data config to extract characteristics
        base_config_name = base_experiment_name.replace('_mlp_001_optimal', '')
        data_config_path = project_root / "configs" / "data_generation" / f"{base_config_name}_config.yml"
        
        if data_config_path.exists():
            from src.data_generator_module import utils as data_utils
            data_config_dict = data_utils.load_yaml_config(data_config_path)
            
            # Extract experiment characteristics
            experiment_characteristics = extract_experiment_characteristics(base_experiment_name, data_config_dict)
            
            # Create a single row for this experiment
            experiment_row = {
                'experiment_family': base_experiment_name,
                'n_samples': experiment_characteristics['n_samples'],
                'n_features': experiment_characteristics['n_features'], 
                'continuous': experiment_characteristics['continuous'],
                'discrete': experiment_characteristics['discrete'],
                'separation': experiment_characteristics['separation'],
                'perturbation': experiment_characteristics['perturbation'],
            }
            
            # Add aggregated metrics (mean and std for each metric)
            for metric in ['Test Loss (BCE)', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']:
                if metric in summary_stats.index:
                    experiment_row[f'{metric}_mean'] = summary_stats.loc[metric, 'mean']
                    experiment_row[f'{metric}_std'] = summary_stats.loc[metric, 'std']
            
            # Global tracking spreadsheet path
            global_spreadsheet_path = reports_root() / "experiment_tracking_master.xlsx"
            
            # Ensure the reports directory exists
            global_spreadsheet_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Load existing data or create new DataFrame
                if global_spreadsheet_path.exists():
                    existing_df = pd.read_excel(global_spreadsheet_path, sheet_name='Master_Tracking')
                    # Remove existing row for this experiment if it exists
                    existing_df = existing_df[existing_df['experiment_family'] != base_experiment_name]
                    # Add the new row
                    updated_df = pd.concat([existing_df, pd.DataFrame([experiment_row])], ignore_index=True)
                else:
                    updated_df = pd.DataFrame([experiment_row])
                
                # Sort by experiment family name for consistent ordering
                updated_df = updated_df.sort_values('experiment_family').reset_index(drop=True)
                
                # Save to Excel
                with pd.ExcelWriter(global_spreadsheet_path, engine='openpyxl') as writer:
                    updated_df.to_excel(writer, sheet_name='Master_Tracking', index=False)
                
                print(f"Global experiment tracking updated: {global_spreadsheet_path}")
                
            except Exception as e:
                print(f"Error updating global tracking spreadsheet: {e}")
        else:
            print(f"Warning: Could not find data config at {data_config_path}")
            
    except Exception as e:
        print(f"Error loading data config for experiment characteristics: {e}")

    # --- 5. Generate Visualisations ---
    from src.utils.report_paths import extract_family_base, experiment_family_path
    
    # Extract the family base from the experiment name
    family_base = extract_family_base(base_experiment_name)
    
    # A. Generate Faceted Line Plots
    plot_data = summary_stats.reset_index().rename(columns={'index': 'Metric'})
    
    # Plot 1: Main metrics
    main_metrics_df = plot_data[plot_data['Metric'] != 'Test Loss (BCE)']

    yerr = bounded_yerr(main_metrics_df['mean'].values,
                     main_metrics_df['std'].values)
    
    plt.errorbar(
        main_metrics_df['Metric'],
        main_metrics_df['mean'],
        yerr=yerr,
        fmt='-o', capsize=5, label='Mean ± Std Dev')
    
    value_labels = [f'{val:.3f}' for val in main_metrics_df['mean'].values]
    add_single_series_labels(
        x_positions=range(len(main_metrics_df)),
        values=main_metrics_df['mean'].values,
        labels_text=value_labels,
        color='black'
    )
    
    title = format_plot_title("Aggregated Performance Metrics for", base_experiment_name)
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha="right")
    
    # Adjust y-axis limits to give labels space
    y_min, y_max = calculate_adaptive_ylimits(
        main_metrics_df['mean'].values, 
        main_metrics_df['std'].values
    )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    apply_decimal_formatters(plt.gca(), precision=3)
    plt.tight_layout()
    
    # Save using family-based structure
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
    # Setup
    orig_opt_path = Path(optimal_config)
    project_root = Path(find_project_root())
    models_dir = project_root / "models"

    # 1. Find base family info from the original config name
    #   E.g. base = n1000_f_init5_cont0_disc5_sep5p1
    orig_base = orig_opt_path.stem  # e.g. n1000_f_init5_cont0_disc5_sep5p1_seed0_mlp_001_optimal
    # Remove _seedX_mlp_001_optimal if present
    base_prefix = re.sub(r'_seed\d+_mlp_001_optimal$', '', orig_base)

    print(f"Scanning for experiment families with base: {base_prefix}")

    # 2. Collect all relevant metric files
    all_metric_files = list(models_dir.glob(f"{base_prefix}*_metrics.json"))
    if not all_metric_files:
        print("No matching metric files found at all.")
        return

    # 3. Discover perturbation family bases
    family_bases = set()
    for mf in all_metric_files:
        fname = mf.stem
        if "_pert_" in fname:
            pert_base = fname.split("_seed")[0]
            family_bases.add(pert_base)
        else:
            family_bases.add(fname.split("_seed")[0])  # original

    # 4. For each unique family base, synthesize the needed dummy config and aggregate
    # Get model and suffix info from the provided optimal_config
    stem_parts = orig_opt_path.stem.split("_")
    # Locate the model suffix (e.g., "_mlp_001_optimal")
    for i, s in enumerate(stem_parts):
        if s.startswith("mlp"):
            model_suffix = "_".join(stem_parts[i:])  # e.g., mlp_001_optimal
            break
    else:
        model_suffix = "mlp_001_optimal"

    for family in sorted(family_bases):
        # synthesize e.g. n1000_f_init5_cont0_disc5_sep5p1_seed0_mlp_001_optimal.yml or
        #            n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed0_mlp_001_optimal.yml
        family_config_filename = f"{family}_seed0_{model_suffix}.yml"
        family_config_path = orig_opt_path.parent / family_config_filename

        # If file doesn't exist, copy the original optimal config—it's only for filename pattern matching.
        if not family_config_path.exists():
            shutil.copyfile(orig_opt_path, family_config_path)

        # Aggregate results for this family
        print(f"\nAggregating family: {family}")
        aggregate(str(family_config_path))

    print("\n--- Auto-aggregation of all original and perturbed experiment families complete. ---")

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