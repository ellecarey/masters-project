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
from .comparison import compare_families


def aggregate(optimal_config: str):
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
    if not optimal_config_path.exists():
        print(f"Error: Optimal config file not found at '{optimal_config_path}'")
        return

    family_match = re.match(r'(.+?)_seed\d+_.*_optimal', optimal_config_path.stem)
    if not family_match:
        print(f"Error: Could not determine experiment family from '{optimal_config_path.name}'.")
        return
    family_base_name = family_match.group(1)
    print(f"--- Aggregating results for experiment family: {family_base_name} ---")

    # --- 2. Find and Load Metric Files ---
    metric_files = list(models_dir.glob(f"{family_base_name}_seed*_metrics.json"))
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

    # --- 4. Generate Standardized Plot Title ---
    data_config_path = project_root / "configs" / "data_generation" / f"{family_base_name}_seed0_config.yml"
    subtitle = f"Family: {family_base_name}"
    if data_config_path.exists():
        data_config_dict = data_utils.load_yaml_config(data_config_path)
        subtitle = generate_subtitle_from_config(data_config_dict)
    else:
        print(f"Warning: Could not find data config at {data_config_path} to generate a detailed plot title.")

    # --- 5. Plotting and Saving ---
    plt.figure()
    plot_data = summary_stats.reset_index().rename(columns={'index': 'Metric'})
    main_metrics_df = plot_data[plot_data['Metric'] != 'Test Loss (BCE)']
    yerr = bounded_yerr(main_metrics_df['mean'].values, main_metrics_df['std'].values)
    plt.errorbar(main_metrics_df['Metric'], main_metrics_df['mean'], yerr=yerr, fmt='-o', capsize=5, label='Mean ± Std Dev')
    
    value_labels = [f'{val:.3f}' for val in main_metrics_df['mean'].values]
    add_single_series_labels(x_positions=range(len(main_metrics_df)), values=main_metrics_df['mean'].values, labels_text=value_labels, color='black')
    
    plt.suptitle("Aggregated Performance Metrics", y=0.98)

    if subtitle:
        y_pos = 0.92  # Starting y-position for the first subtitle line
        for line in subtitle.split('\n'):
            # Use figtext to add each line as centered text on the FIGURE.
            # Font settings are now inherited from your plot style.
            plt.figtext(0.5, y_pos, line, ha='center', wrap=True)
            y_pos -= 0.05  # Move down for the next line
    
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha="right")
    plt.yticks()
    
    y_min, y_max = calculate_adaptive_ylimits(main_metrics_df['mean'].values, main_metrics_df['std'].values)
    plt.grid(True, linestyle='--', alpha=0.7)
    apply_decimal_formatters(plt.gca(), precision=3)
    
    # 3. Adjust layout to make room for our custom titles.
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
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
        print(
            f"Error: The provided optimal config file does not exist: {orig_opt_path}"
        )
        return

    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    orig_base_name = orig_opt_path.stem
    base_prefix = re.sub(r'_(?:seed\d+|training)_mlp_.*_optimal$', '', orig_base_name)
    print(f"Scanning for experiment families with base: {base_prefix}")
    all_metric_files = list(models_dir.glob(f"{base_prefix}*_metrics.json"))
    if not all_metric_files:
        print("No matching metric files found.")
        return

    family_bases = {
        mf.stem.split("_seed")[0] for mf in all_metric_files if "_seed" in mf.stem
    }
    model_suffix_match = re.search(r'(_mlp_.*_optimal)$', orig_opt_path.stem)
    if not model_suffix_match:
        print(f"Error: Could not determine model suffix from '{orig_opt_path.name}'")
        return
    model_suffix = model_suffix_match.group(1)

    for family in sorted(family_bases):
        family_config_filename = f"{family}_seed0{model_suffix}.yml"
        family_config_path = orig_opt_path.parent / family_config_filename
        if not family_config_path.exists():
            print(f"Creating temporary config '{family_config_path.name}' for aggregation.")
            shutil.copyfile(orig_opt_path, family_config_path)
        print(f"\nAggregating family: {family}")
        aggregate(str(family_config_path))

    print(
        "\n--- Auto-aggregation of all original and perturbed experiment families complete. ---"
    )

    perturbation_tag = None
    for family in family_bases:
        if "_pert_" in family:
            match = re.search(r'_(pert_.*)$', family)
            if match:
                perturbation_tag = match.group(1)
                break

    if perturbation_tag:
        print("\n" + "=" * 80)
        print("Next Step: Compare the aggregated results")
        print("=" * 80 + "\n")
        print(
            "Use the 'compare-families' command to generate plots comparing the performance"
        )
        print("of the original dataset against the perturbed version.")
        original_family_optimal_config = (
            orig_opt_path.parent / f"{base_prefix}_seed0{model_suffix}.yml"
        )
        compare_command = (
            f"uv run experiment_manager.py compare-families \\\n"
            f" --original-optimal-config {original_family_optimal_config} \\\n"
            f" --perturbation-tag {perturbation_tag}"
        )
        print(compare_command)
        print("\n" + "=" * 80)


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
        "--optimal-config",
        required=True,
        type=str,
        help="Path to one of the final optimal config files used for the runs (e.g., ..._optimal.yml).",
    )

    # Compare-families
    comp = subparsers.add_parser(
        "compare-families", help="Compare original and perturbed experiment families."
    )
    comp.add_argument(
        "--original-optimal-config",
        required=True,
        type=str,
        help="Original (unperturbed) optimal config YAML file.",
    )
    comp.add_argument(
        "--perturbation-tag",
        required=True,
        type=str,
        help="Perturbation tag used in filenames for the perturbed family (e.g. 'pert_f4n_by1p0s').",
    )

    # Fully auto aggregation (all families discovered by pattern)
    agg_all = subparsers.add_parser(
        "aggregate-all",
        help="Aggregate original and all detected perturbed experiment families automatically.",
    )
    agg_all.add_argument(
        "--optimal-config",
        required=True,
        type=str,
        help="Path to the optimal config file of the ORIGINAL family.",
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
