from .aggregation import aggregate_family_results
import pandas as pd
import re
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from src.utils.filenames import metrics_filename, model_filename, config_filename, parse_optimal_config_name
from src.data_generator_module.utils import find_project_root
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.utils.report_paths import artefact_path, reports_root
from src.utils.plotting_helpers import bounded_yerr, calculate_adaptive_ylimits, add_smart_value_labels, add_single_series_labels, format_plot_title

TRAINING_SEED = 99

def compare_families(original_optimal_config, perturbation_tag):
    apply_custom_plot_style()
    orig_config_path = Path(original_optimal_config)
    project_root = Path(find_project_root())

    # --- 1. Identify experiment names and discover seeds dynamically ---
    dataset_base, model_name, _, _ = parse_optimal_config_name(orig_config_path)
    
    models_dir = project_root / "models"
    orig_pattern = metrics_filename(dataset_base, model_name, seed="*", perturbation_tag=None)
    orig_metric_files = list(models_dir.glob(orig_pattern.replace('*', '[0-9]*')))
    
    if not orig_metric_files:
        print(f"Error: No metric files found for original family.")
        return

    all_seeds = sorted([int(re.search(r'_seed(\d+)', f.stem).group(1)) for f in orig_metric_files])
    seeds = [s for s in all_seeds if s != TRAINING_SEED]
    print(f"Discovered and using {len(seeds)} evaluation seeds for comparison: {seeds}")

    # --- 2. Gather Metrics ---
    orig_metrics_files = [
        models_dir / metrics_filename(dataset_base, model_name, seed=s, perturbation_tag=None)
        for s in seeds
    ]
    pert_metrics_files = [
        models_dir / metrics_filename(dataset_base, model_name, seed=s, perturbation_tag=perturbation_tag)
        for s in seeds
    ]

    orig_metrics_data = []
    for file_path in orig_metrics_files:
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['seed'] = int(re.search(r'_seed(\d+)', file_path.stem).group(1))
                orig_metrics_data.append(data)
        else:
            print(f"Warning: Metric file not found: {file_path}")

    pert_metrics_data = []
    for file_path in pert_metrics_files:
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['seed'] = int(re.search(r'_seed(\d+)', file_path.stem).group(1))
                pert_metrics_data.append(data)
        else:
            print(f"Warning: Metric file not found: {file_path}")

    # Create DataFrames from the loaded data
    orig_df = pd.DataFrame(orig_metrics_data)
    pert_df = pd.DataFrame(pert_metrics_data)

    if orig_df.empty or pert_df.empty:
        print("Error: Could not load results for one or both families.")
        return

    merged = pd.merge(orig_df, pert_df, on='seed', suffixes=('_orig', '_pert'))
    all_metrics = [col for col in orig_df.columns if col not in ['seed', 'Test Loss (BCE)']]
    metrics = all_metrics
    for m in metrics:
        merged[f'{m}_delta'] = merged[f"{m}_pert"] - merged[f"{m}_orig"]
    
    orig_summary = orig_df[metrics].agg(['mean', 'std'])
    pert_summary = pert_df[metrics].agg(['mean', 'std'])

    orig_name = f"{dataset_base}"
    pert_name = f"{dataset_base}_{perturbation_tag}"
    comparison_csv_path = artefact_path(
        experiment=f"{orig_name}_vs_{pert_name}",
        art_type="comparison",
        filename=f"comparison_{orig_name}_vs_{pert_name}.csv"
    )
    merged.to_csv(comparison_csv_path, index=False)
    print(f"Comparison data saved to: {comparison_csv_path}")

    print("\nComparison table:")
    print(merged.to_string(index=False))
    print("\nSummary (mean/std):")
    print(merged[[f'{m}_delta' for m in metrics]].agg(['mean', 'std']))

    # ... (Plotting logic is also unchanged)
    plt.figure()
    x_pos = range(len(metrics))
    all_means = np.concatenate([orig_summary.loc['mean'].values, pert_summary.loc['mean'].values])
    all_stds = np.concatenate([orig_summary.loc['std'].values, pert_summary.loc['std'].values])
    data_min = np.min(all_means - all_stds)
    data_max = np.max(all_means + all_stds)
    if data_min > 0.9:
        y_min, y_max = max(0.9, data_min - 0.01), min(1.02, data_max + 0.01)
    else:
        y_min, y_max = calculate_adaptive_ylimits(all_means, all_stds)

    orig_yerr = bounded_yerr(orig_summary.loc['mean'].values, orig_summary.loc['std'].values)
    pert_yerr = bounded_yerr(pert_summary.loc['mean'].values, pert_summary.loc['std'].values)

    plt.errorbar(x_pos, orig_summary.loc['mean'], yerr=orig_yerr, marker='o', linewidth=2, capsize=5, capthick=2, label='Original', color='blue', markersize=8)
    plt.errorbar(x_pos, pert_summary.loc['mean'], yerr=pert_yerr, marker='s', linewidth=2, capsize=5, capthick=2, label='Perturbed', color='red', markersize=8)
    
    comp_title = format_plot_title("Performance Comparison: Original vs Perturbed", f"{orig_name} vs {pert_name}")
    plt.title(comp_title)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x_pos, metrics, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    orig_values = orig_summary.loc['mean'].values
    pert_values = pert_summary.loc['mean'].values
    orig_labels = [f'{val:.3f}' for val in orig_values]
    pert_labels = [f'{val:.3f}' for val in pert_values]

    add_smart_value_labels(x_positions=x_pos, values1=orig_values, values2=pert_values, labels1_text=orig_labels, labels2_text=pert_labels, color1='blue', color2='red')
    
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    comparison_plot_path = artefact_path(experiment=f"{orig_name}_vs_{pert_name}", art_type="comparison", filename=f"{orig_name}_vs_{pert_name}_metrics_comparison.pdf")
    plt.savefig(comparison_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison line plot: {comparison_plot_path}")

    # ... (delta plot logic)
    deltas_mean = merged[[f'{m}_delta' for m in metrics]].mean()
    deltas_std = merged[[f'{m}_delta' for m in metrics]].std()
    delta_min_val, delta_max_val = np.min(deltas_mean - deltas_std), np.max(deltas_mean + deltas_std)
    padding = (delta_max_val - delta_min_val) * 0.1
    y_min_delta, y_max_delta = delta_min_val - padding, delta_max_val + padding
    
    plt.figure()
    plt.errorbar(x_pos, deltas_mean, yerr=deltas_std, marker='o', linewidth=2, capsize=5, capthick=2, color='green', markersize=8)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.ylim(y_min_delta, y_max_delta)
    delta_title = format_plot_title("Performance Difference (Perturbed - Original)", f"{orig_name} vs {pert_name}")
    plt.title(delta_title)
    plt.xlabel('Metrics')
    plt.ylabel('Difference in Score')
    plt.xticks(x_pos, metrics, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    delta_labels = [f'{val:.3f}' for val in deltas_mean]
    add_single_series_labels(x_positions=x_pos, values=deltas_mean, labels_text=delta_labels, color='green')
    
    plt.tight_layout()
    delta_plot_path = artefact_path(experiment=f"{orig_name}_vs_{pert_name}", art_type="comparison", filename=f"{orig_name}_vs_{pert_name}_delta_comparison.pdf")
    plt.savefig(delta_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved delta comparison plot: {delta_plot_path}")

    print(f"\nAll comparison artifacts saved under: {reports_root() / 'comparisons' / f'{orig_name}_vs_{pert_name}'}")