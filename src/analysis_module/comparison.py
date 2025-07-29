from .aggregation import aggregate_family_results
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.filenames import metrics_filename, model_filename, config_filename, parse_optimal_config_name
from src.data_generator_module.utils import find_project_root
from src.utils.report_paths import artefact_path, reports_root

def compare_families(original_optimal_config, perturbation_tag):
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
    metrics = ['Accuracy', 'AUC', 'F1-Score']
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

    # Generate single line plot comparison (similar to training history)
    orig_summary = orig_df[metrics].agg(['mean', 'std'])
    pert_summary = pert_df[metrics].agg(['mean', 'std'])
    
    # Create figure with similar style to training history
    plt.figure(figsize=(12, 8))
    
    # X-axis positions for metrics
    x_pos = range(len(metrics))
    
    # Plot lines for original and perturbed results
    plt.errorbar(x_pos, orig_summary.loc['mean'], yerr=orig_summary.loc['std'], 
                 marker='o', linewidth=2, capsize=5, capthick=2, 
                 label='Original', color='blue', markersize=8)
    
    plt.errorbar(x_pos, pert_summary.loc['mean'], yerr=pert_summary.loc['std'], 
                 marker='s', linewidth=2, capsize=5, capthick=2, 
                 label='Perturbed', color='red', markersize=8)
    
    # Customize the plot
    plt.title(f'Performance Comparison: Original vs Perturbed\n({orig_name} vs {pert_name})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x_pos, metrics, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Add value labels on points
    for i, metric in enumerate(metrics):
        orig_val = orig_summary.loc['mean', metric]
        pert_val = pert_summary.loc['mean', metric]
        
        # Add labels above the points
        plt.annotate(f'{orig_val:.4f}', 
                    (i, orig_val), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=10, color='blue')
        
        plt.annotate(f'{pert_val:.4f}', 
                    (i, pert_val), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center', fontsize=10, color='red')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save using the new report paths system
    comparison_plot_path = artefact_path(
        experiment=f"{orig_name}_vs_{pert_name}",
        art_type="comparison",
        filename=f"{orig_name}_vs_{pert_name}_metrics_comparison.pdf"
    )
    
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison line plot: {comparison_plot_path}")

    # Optional: Create a second plot showing the deltas (differences)
    deltas_mean = merged[[f'{m}_delta' for m in metrics]].mean()
    deltas_std = merged[[f'{m}_delta' for m in metrics]].std()
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_pos, deltas_mean, yerr=deltas_std, 
                 marker='o', linewidth=2, capsize=5, capthick=2, 
                 color='green', markersize=8)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title(f'Performance Difference (Perturbed - Original)\n({orig_name} vs {pert_name})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Difference in Score', fontsize=14)
    plt.xticks(x_pos, metrics, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, (metric, delta) in enumerate(zip(metrics, deltas_mean)):
        plt.annotate(f'{delta:.4f}', 
                    (i, delta), 
                    textcoords="offset points", 
                    xytext=(0,10 if delta >= 0 else -15), 
                    ha='center', fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save delta plot
    delta_plot_path = artefact_path(
        experiment=f"{orig_name}_vs_{pert_name}",
        art_type="comparison",
        filename=f"{orig_name}_vs_{pert_name}_delta_comparison.pdf"
    )
    
    plt.savefig(delta_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved delta comparison plot: {delta_plot_path}")

    print(f"\nAll comparison artifacts saved under: {reports_root() / 'comparisons' / f'{orig_name}_vs_{pert_name}'}")