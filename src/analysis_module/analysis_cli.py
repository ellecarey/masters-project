import argparse
import shutil
import re
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.data_generator_module import utils as data_utils
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.utils.report_paths import experiment_family_path
from src.utils.plotting_helpers import (
    bounded_yerr,
    calculate_adaptive_ylimits,
    add_single_series_labels,
    format_plot_title,
    apply_decimal_formatters,
    generate_subtitle_from_config,
)
from src.data_generator_module.utils import find_project_root
from src.analysis_module.global_tracker import generate_global_tracking_sheet
from .comparison import compare_families

def aggregate(optimal_config: str, family_base_name_override: str = None):
    """
    Aggregates results from multi-seed training runs, saves them to a
    spreadsheet, and generates summary plots with standardized, non-overlapping titles.
    """
    apply_custom_plot_style()
    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    
     # --- 1. Identify the Experiment Family ---
    optimal_config_path = Path(optimal_config)
    if not optimal_config_path.is_absolute():
        optimal_config_path = project_root / optimal_config

    # Extract model name robustly from the optimal config file
    model_name_match = re.search(r'_(mlp_\d+)_optimal', optimal_config_path.stem)
    if not model_name_match:
        print(f"Error: Could not determine model name from '{optimal_config_path.name}'.")
        return
    model_name = model_name_match.group(1)

    if family_base_name_override:
        family_base_name = family_base_name_override
    else:
        optimal_config_path = Path(optimal_config)
        if not optimal_config_path.is_absolute():
            optimal_config_path = project_root / optimal_config
        if not optimal_config_path.exists():
            print(f"Error: Optimal config file not found at '{optimal_config_path}'")
            return
        family_match = re.match(r'(.+?)_seed\d+_.*_optimal', optimal_config_path.stem)
        if not family_match:
            family_match = re.match(r'(.+?)_training_.*_optimal', optimal_config_path.stem)
            if not family_match:
                print(f"Error: Could not determine experiment family from '{optimal_config_path.name}'.")
                return
        family_base_name = family_match.group(1)
        
    print(f"--- Aggregating results for experiment family: {family_base_name} ---")

    # --- 2. Find and Load Metric Files ---
    metric_files = list(models_dir.glob(f"{family_base_name}_seed*_{model_name}_metrics.json"))
    if not metric_files:
        print(f"Error: No metric files found for family '{family_base_name}' in '{models_dir}'.")
        return
    print(f"Found {len(metric_files)} metric files to aggregate.")
    all_metrics = []
    for f in metric_files:
        match = re.search(r'_seed(\d+)', f.stem)
        if match:
            try:
                seed = int(match.group(1))
                with open(f, 'r') as file:
                    metrics = json.load(file)
                    metrics['seed'] = seed
                    all_metrics.append(metrics)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping file '{f.name}' due to a processing error: {e}")

    if not all_metrics:
        print("Error: No valid metric files could be processed.")
        return

    # --- 3. Process and Display Results ---
    results_df = pd.DataFrame(all_metrics).sort_values('seed').reset_index(drop=True)
    summary_stats = results_df.drop(columns=['seed']).agg(['mean', 'std']).T
    summary_stats['mean'] = summary_stats['mean'].round(4)
    summary_stats['std'] = summary_stats['std'].round(4)
    print("\n--- Aggregated Results ---")
    print(results_df.round(4).to_string(index=False))
    print("\n--- Summary Statistics (Mean & Std Dev) ---")
    print(summary_stats)

    # Save the summary statistics DataFrame to a dedicated spreadsheet
    spreadsheet_save_path = experiment_family_path(
        full_experiment_name=family_base_name,
        art_type="spreadsheet",
        subfolder=family_base_name,
        filename=f"{family_base_name}_summary.csv"
    )
    summary_stats.to_csv(spreadsheet_save_path)
    print(f"\nSaved summary spreadsheet to: {spreadsheet_save_path}")

    # --- 4. Generate Standardized Plot Title ---
    data_config_path = project_root / "configs" / "data_generation" / f"{family_base_name}_seed0_config.yml"
    subtitle = f"Family: {family_base_name}"
    if data_config_path.exists():
        data_config_dict = data_utils.load_yaml_config(data_config_path)
        subtitle = generate_subtitle_from_config(data_config_dict)
    else:
        print(f"Warning: Could not find data config at {data_config_path} to generate a detailed plot title.")

    # --- 5. Plotting and Saving ---
    fig, ax = plt.subplots()
    plot_data = summary_stats.reset_index().rename(columns={'index': 'Metric'})
    main_metrics_df = plot_data[plot_data['Metric'] != 'Test Loss (BCE)']
    yerr = bounded_yerr(main_metrics_df['mean'].values, main_metrics_df['std'].values)
    ax.errorbar(main_metrics_df['Metric'], main_metrics_df['mean'], yerr=yerr, fmt='-o', capsize=5, label='Mean ± Std Dev')
    value_labels = [f'{val:.3f}' for val in main_metrics_df['mean'].values]
    add_single_series_labels(
        ax=ax,
        x_positions=range(len(main_metrics_df)),
        values=main_metrics_df['mean'].values,
        labels_text=value_labels,
        color='black'
    )
    fig.suptitle("Aggregated Performance Metrics", y=0.98)
    if subtitle:
        y_pos = 0.92
        for line in subtitle.split('\n'):
            fig.text(0.5, y_pos, line, ha='center', wrap=True)
            y_pos -= 0.05
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_xticks(range(len(main_metrics_df['Metric'])))
    ax.set_xticklabels(main_metrics_df['Metric'], rotation=45, ha="right")
    y_min, y_max = calculate_adaptive_ylimits(main_metrics_df['mean'].values, main_metrics_df['std'].values)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle='--', alpha=0.7)
    apply_decimal_formatters(ax, precision=3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])

    plot_save_path = experiment_family_path(
        full_experiment_name=family_base_name,
        art_type="figure",
        subfolder="aggregation_summary",
        filename=f"{family_base_name}_summary.pdf"
    )
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()

    print(f"\nSaved main metrics plot to: {plot_save_path}")
    print("\n--- Aggregation complete. ---")

def aggregate_all_families(optimal_config: str):
    """
    Aggregate both the original and all detected perturbed experiment families
    sharing the same base as the given optimal config, fully automatically.
    """
    orig_opt_path = Path(optimal_config)
    if not orig_opt_path.exists():
        print(f"Error: The provided optimal config file does not exist: {orig_opt_path}")
        return
    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    configs_gen_dir = project_root / "configs/training/generated"
    orig_base_name = orig_opt_path.stem
    base_prefix_match = re.match(r'^(n\d+_f_init\d+_cont\d+_disc\d+_sep[\d_p]+)', orig_opt_path.stem)
    if not base_prefix_match:
        print(f"Error: Could not determine the base prefix from '{orig_opt_path.name}'")
        return
    base_prefix = base_prefix_match.group(1)
    # Extract the model suffix (e.g., _mlp_001_optimal)
    model_suffix_match = re.search(r'(_mlp_.*_optimal)$', orig_opt_path.stem)
    if not model_suffix_match:
        print(f"Error: Could not determine model suffix from '{orig_opt_path.name}'")
        return
    model_suffix = model_suffix_match.group(1)
    print(f"Scanning for experiment families with base: {base_prefix}")
    all_metric_files = list(models_dir.glob(f"{base_prefix}*_metrics.json"))
    if not all_metric_files:
        print("No matching metric files found.")
        return
    # Find all unique families (including original and perturbed)
    family_bases = {mf.stem.split("_seed")[0].rstrip('_') for mf in all_metric_files if "_seed" in mf.stem}

    for family in sorted(family_bases):
        print(f"\nAggregating family: {family}")
        # Pass the family name to the aggregate function
        aggregate(optimal_config=str(orig_opt_path), family_base_name_override=family)

    print("\n--- Auto-aggregation of all original and perturbed experiment families complete. ---")
    print("\n--- Automatically generating global tracking spreadsheet... ---")
    generate_global_tracking_sheet()

    # Automatically run comparison if a perturbed family was found
    perturbation_tags = []
    for family in family_bases:
        if "_pert_" in family:
            match = re.search(r'_pert_.*', family)
            if match:
                perturbation_tags.append(match.group(0))
    # Generate comparison for each perturbation
    for perturbation_tag in set(perturbation_tags): # Use set to avoid duplicates
        print(f"\n{'='*80}")
        print(f"Generating comparison plots for: {perturbation_tag}")
        print("="*80 + "\n")
        original_family_optimal_config = configs_gen_dir / f"{base_prefix}_seed0{model_suffix}.yml"
        compare_families(
            original_optimal_config=str(original_family_optimal_config),
            perturbation_tag=perturbation_tag
        )

def main():
    parser = argparse.ArgumentParser(description="CLI tools for experiment analysis (aggregation and comparison).")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    agg = subparsers.add_parser("aggregate", help="Aggregate multi-seed results for an experiment family.")
    agg.add_argument("--optimal-config", required=True, type=str, help="Path to one of the final optimal config files used for the runs (e.g., ..._optimal.yml).")
    
    comp = subparsers.add_parser("compare-families", help="Compare original and perturbed experiment families.")
    comp.add_argument("--original-optimal-config", required=True, type=str, help="Original (unperturbed) optimal config YAML file.")
    comp.add_argument("--perturbation-tag", required=True, type=str, help="Perturbation tag used in filenames for the perturbed family (e.g. 'pert_f4n_by1p0s').")
    
    agg_all = subparsers.add_parser("aggregate-all", help="Aggregate original and all detected perturbed experiment families automatically.")
    agg_all.add_argument("--optimal-config", required=True, type=str, help="Path to the optimal config file of the ORIGINAL family.")
    
    args = parser.parse_args()

    if args.command == "aggregate":
        aggregate(args.optimal_config)
    elif args.command == "compare-families":
        # Note: This import is kept for direct CLI use, but the automated call above handles the pipeline
        from .comparison import compare_families
        compare_families(args.original_optimal_config, args.perturbation_tag)
    elif args.command == "aggregate-all":
        aggregate_all_families(args.optimal_config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
