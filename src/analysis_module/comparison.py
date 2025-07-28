from .aggregation import aggregate_family_results
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.filenames import metrics_filename, model_filename, config_filename, parse_optimal_config_name
from src.data_generator_module.utils import find_project_root
from src.utils.report_paths import artefact_path

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
    
    # 4. Print summary and plot
    print("\nComparison table:")
    print(merged.to_string(index=False))
    print("\nSummary (mean/std):")
    print(merged[[f'{m}_delta' for m in metrics]].agg(['mean', 'std']))

    # Generate comparison plots using new system
    for m in metrics:
        means = [orig_summary.loc['mean', m], pert_summary.loc['mean', m]]
        stds = [orig_summary.loc['std', m], pert_summary.loc['std', m]]
        
        plt.figure()
        plt.bar(['Original', 'Perturbed'], means, yerr=stds, capsize=4)
        plt.title(f'{m}: Original vs Perturbed')
        plt.ylabel(m)
        plt.tight_layout()
        
        # Save using new report paths system
        plot_path = artefact_path(
            experiment=f"{orig_name}_vs_{pert_name}",
            art_type="comparison", 
            filename=f"{orig_name}_vs_{pert_name}_{m}.pdf"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved comparison plot: {plot_path}")

    print(f"\nAll comparison artifacts saved under: {reports_root() / 'comparisons' / f'{orig_name}_vs_{pert_name}'}")
