from .aggregation import aggregate_family_results
import pandas as pd
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.filenames import metrics_filename, model_filename, config_filename, parse_optimal_config_name
from src.data_generator_module.utils import find_project_root
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.utils.report_paths import artefact_path, reports_root
from src.utils.plotting_helpers import bounded_yerr, calculate_adaptive_ylimits, add_smart_value_labels, add_single_series_labels, format_plot_title


def compare_families(original_optimal_config, perturbation_tag):
    apply_custom_plot_style()
    orig_config_path = Path(original_optimal_config)
    # Parse components from filename or config
    dataset_base, model_name, seed, pert_tag = parse_optimal_config_name(orig_config_path)
    seeds = [0, 1, 2, 3, 4]  # or discover programmatically
    orig_name = f"{dataset_base}"
    pert_name = f"{dataset_base}_{perturbation_tag}"
    
    # Gather metrics filenames for all seeds
    project_root = Path(find_project_root())
    orig_metrics_files = [
        project_root / "models" / metrics_filename(dataset_base, model_name, seed=s, perturbation_tag=None)
        for s in seeds
    ]
    pert_metrics_files = [
        project_root / "models" / metrics_filename(dataset_base, model_name, seed=s, perturbation_tag=perturbation_tag)
        for s in seeds
    ]
    
    orig_metrics = []
    for f in orig_metrics_files:
        if f.exists():
            with open(f) as fh:
                row = pd.read_json(fh, typ='series')
                row['seed'] = int(re.search(r'seed(\d+)', f.stem).group(1))
                orig_metrics.append(row)
    
    pert_metrics = []
    for f in pert_metrics_files:
        if f.exists():
            with open(f) as fh:
                row = pd.read_json(fh, typ='series')
                row['seed'] = int(re.search(r'seed(\d+)', f.stem).group(1))
                pert_metrics.append(row)
    
    # DataFrames
    orig_df = pd.DataFrame(orig_metrics)
    pert_df = pd.DataFrame(pert_metrics)
    if orig_df.empty or pert_df.empty:
        print("Error: Could not load results for one or both families.")
        return

    # 3. Merge and compare by seed
    merged = pd.merge(orig_df, pert_df, on='seed', suffixes=('_orig', '_pert'))
    
    # Include all metrics except Test Loss (BCE) ***
    all_metrics = [col for col in orig_df.columns if col not in ['seed', 'Test Loss (BCE)']]
    metrics = all_metrics  # Now includes: Accuracy, F1-Score, Precision, Recall, AUC
    
    for m in metrics:
        merged[f'{m}_delta'] = merged[f"{m}_pert"] - merged[f"{m}_orig"]
    
    orig_summary = orig_df[metrics].agg(['mean', 'std'])
    pert_summary = pert_df[metrics].agg(['mean', 'std'])
    
    # Save comparison as CSV using new report paths system
    comparison_csv_path = artefact_path(
        experiment=f"{orig_name}_vs_{pert_name}",
        art_type="comparison",
        filename=f"comparison_{orig_name}_vs_{pert_name}.csv"
    )
    merged.to_csv(comparison_csv_path, index=False)
    print(f"Comparison data saved to: {comparison_csv_path}")
    
    # 4. Print summary and generate single line plot comparison
    print("\nComparison table:")
    print(merged.to_string(index=False))
    print("\nSummary (mean/std):")
    print(merged[[f'{m}_delta' for m in metrics]].agg(['mean', 'std']))

    # Generate single line plot comparison 
    plt.figure()
    
    # X-axis positions for metrics
    x_pos = range(len(metrics))
    
    all_means = np.concatenate([orig_summary.loc['mean'].values, 
                               pert_summary.loc['mean'].values])
    all_stds = np.concatenate([orig_summary.loc['std'].values, 
                              pert_summary.loc['std'].values])
    
    # Calculate tighter, more appropriate y-limits for high-performing metrics
    data_min = np.min(all_means - all_stds)
    data_max = np.max(all_means + all_stds)
    
    # For metrics clustered near 1.0, use a much tighter range
    if data_min > 0.9:  # High-performing metrics
        y_min = max(0.9, data_min - 0.01)  # Small padding
        y_max = min(1.02, data_max + 0.01)  # Small padding, cap at 1.02
    else:
        # Use adaptive limits for wider ranges
        y_min, y_max = calculate_adaptive_ylimits(all_means, all_stds)
    
    # Plot lines for original and perturbed results
    orig_yerr = bounded_yerr(orig_summary.loc['mean'].values,
                            orig_summary.loc['std'].values)
    
    pert_yerr = bounded_yerr(pert_summary.loc['mean'].values,
                            pert_summary.loc['std'].values)
    
    plt.errorbar(x_pos, orig_summary.loc['mean'], yerr=orig_yerr,
                 marker='o', linewidth=2, capsize=5, capthick=2,
                 label='Original', color='blue', markersize=8)
    
    plt.errorbar(x_pos, pert_summary.loc['mean'], yerr=pert_yerr,
                 marker='s', linewidth=2, capsize=5, capthick=2,
                 label='Perturbed', color='red', markersize=8)
    
    # Customize the plot
    comp_title = format_plot_title("Performance Comparison: Original vs Perturbed", f"{orig_name} vs {pert_name}")
    plt.title(comp_title)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x_pos, metrics, rotation=45, ha='right')  
    plt.yticks()
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Prepare label data
    orig_values = orig_summary.loc['mean'].values
    pert_values = pert_summary.loc['mean'].values
    orig_labels = [f'{val:.3f}' for val in orig_values]
    pert_labels = [f'{val:.3f}' for val in pert_values]
    
    # Add smart labels
    add_smart_value_labels(
        x_positions=x_pos,
        values1=orig_values,
        values2=pert_values,
        labels1_text=orig_labels,
        labels2_text=pert_labels,
        color1='blue',
        color2='red',
    )
        
        # Set the y-limits AFTER all plot elements
    plt.ylim(y_min, y_max)
        
    plt.tight_layout()
    
    # Save using the new report paths system
    comparison_plot_path = artefact_path(
        experiment=f"{orig_name}_vs_{pert_name}",
        art_type="comparison",
        filename=f"{orig_name}_vs_{pert_name}_metrics_comparison.pdf"
    )
    
    plt.savefig(comparison_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison line plot: {comparison_plot_path}")

    deltas_mean = merged[[f'{m}_delta' for m in metrics]].mean()
    deltas_std = merged[[f'{m}_delta' for m in metrics]].std()
    
    # Calculate adaptive limits for deltas 
    delta_min = np.min(deltas_mean - deltas_std)
    delta_max = np.max(deltas_mean + deltas_std)
    delta_range = delta_max - delta_min
    padding = delta_range * 0.1
    
    y_min_delta = delta_min - padding
    y_max_delta = delta_max + padding
    
    plt.figure()  
    plt.errorbar(x_pos, deltas_mean, yerr=deltas_std,
                 marker='o', linewidth=2, capsize=5, capthick=2,
                 color='green', markersize=8)
        
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.ylim(y_min_delta, y_max_delta)

    delta_title = format_plot_title("Performance Difference (Perturbed - Original)", f"{orig_name} vs {pert_name}")
    plt.title(delta_title)
    plt.xlabel('Metrics')
    plt.ylabel('Difference in Score')
    plt.xticks(x_pos, metrics, rotation=45, ha='right')  
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    delta_labels = [f'{val:.3f}' for val in deltas_mean]
    add_single_series_labels(
        x_positions=x_pos,
        values=deltas_mean,
        labels_text=delta_labels,
        color='green',
    )
    
    plt.tight_layout()
    
    # Save delta plot
    delta_plot_path = artefact_path(
        experiment=f"{orig_name}_vs_{pert_name}",
        art_type="comparison",
        filename=f"{orig_name}_vs_{pert_name}_delta_comparison.pdf"
    )
    
    plt.savefig(delta_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved delta comparison plot: {delta_plot_path}")

    print(f"\nAll comparison artifacts saved under: {reports_root() / 'comparisons' / f'{orig_name}_vs_{pert_name}'}")
