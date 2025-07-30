from .aggregation import aggregate_family_results
import pandas as pd
import re
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

from src.data_generator_module import utils as data_utils
from src.utils.filenames import metrics_filename, model_filename, config_filename, parse_optimal_config_name
from src.data_generator_module.utils import find_project_root
from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.utils.report_paths import artefact_path, reports_root
from src.utils.plotting_helpers import bounded_yerr, calculate_adaptive_ylimits, add_smart_value_labels, add_single_series_labels, format_plot_title, generate_subtitle_from_config

TRAINING_SEED = 99
def compare_families(original_optimal_config, perturbation_tag):
    """
    Compares original and perturbed experiment families, generating plots
    with dynamic titles and data labels.
    """
    apply_custom_plot_style()
    orig_config_path = Path(original_optimal_config)
    project_root = Path(find_project_root())

    # --- 1. & 2. Data Loading (No changes) ---
    dataset_base, model_name, _, _ = parse_optimal_config_name(orig_config_path)
    models_dir = project_root / "models"
    orig_pattern = metrics_filename(dataset_base, model_name, seed="*", perturbation_tag=None)
    orig_metric_files = list(models_dir.glob(orig_pattern.replace('*', '[0-9]*')))
    if not orig_metric_files:
        print(f"Error: No metric files found for original family '{dataset_base}'.")
        return
    all_seeds = sorted([int(re.search(r'_seed(\d+)', f.stem).group(1)) for f in orig_metric_files])
    seeds = [s for s in all_seeds if s != TRAINING_SEED]
    print(f"Discovered and using {len(seeds)} evaluation seeds for comparison: {seeds}")

    orig_metrics_files = [models_dir / metrics_filename(dataset_base, model_name, seed=s) for s in seeds]
    pert_metrics_files = [models_dir / metrics_filename(dataset_base, model_name, seed=s, perturbation_tag=perturbation_tag) for s in seeds]
    orig_metrics_data = [json.loads(f.read_text()) for f in orig_metrics_files if f.exists()]
    pert_metrics_data = [json.loads(f.read_text()) for f in pert_metrics_files if f.exists()]
    if not orig_metrics_data or not pert_metrics_data:
        print("Error: Metric files are missing. Cannot create comparison.")
        return
    orig_df = pd.DataFrame(orig_metrics_data)
    orig_df['seed'] = seeds
    pert_df = pd.DataFrame(pert_metrics_data)
    pert_df['seed'] = seeds
    
    # --- 3. Generate Dynamic Title from Perturbation Config (No changes) ---
    pert_family_base = f"{dataset_base}_{perturbation_tag}"
    pert_data_config_path = project_root / "configs" / "data_generation" / f"{pert_family_base}_seed0_config.yml"
    
    pert_title_part = "Perturbed" 
    if pert_data_config_path.exists():
        pert_config = data_utils.load_yaml_config(str(pert_data_config_path))
        pert_settings = pert_config.get("perturbation_settings")
        if pert_settings and isinstance(pert_settings, list) and pert_settings:
            pert_descs = []
            for p in pert_settings:
                feature = p.get('feature', 'N/A')
                class_label = 'Noise' if p.get('class_label') == 0 else 'Signal'
                sigma_shift = p.get('sigma_shift', 0.0)
                pert_descs.append(f"{feature} ({class_label}) by {sigma_shift:+.1f}σ")
            pert_title_part = "; ".join(pert_descs)

    orig_data_config_path = project_root / "configs" / "data_generation" / f"{dataset_base}_seed0_config.yml"
    pert_family_base = f"{dataset_base}_{perturbation_tag}"
    pert_data_config_path = project_root / "configs" / "data_generation" / f"{pert_family_base}_seed0_config.yml"

    orig_subtitle = "Original:\n" + (generate_subtitle_from_config(data_utils.load_yaml_config(orig_data_config_path)) if orig_data_config_path.exists() else "Details unavailable")
    pert_subtitle = "Perturbed:\n" + (generate_subtitle_from_config(data_utils.load_yaml_config(pert_data_config_path)) if pert_data_config_path.exists() else "Details unavailable")

    # --- 4. Plotting with corrected function calls ---
    merged = pd.merge(orig_df, pert_df, on='seed', suffixes=('_orig', '_pert'))
    metrics = [col for col in orig_df.columns if col not in ['seed', 'Test Loss (BCE)']]
    x_pos = np.arange(len(metrics))

    # --- Main comparison plot ---
    plt.figure()
    orig_summary = orig_df[metrics].agg(['mean', 'std'])
    pert_summary = pert_df[metrics].agg(['mean', 'std'])
    plt.errorbar(x_pos, orig_summary.loc['mean'], yerr=bounded_yerr(orig_summary.loc['mean'], orig_summary.loc['std']), marker='o', label='Original', capsize=5)
    plt.errorbar(x_pos, pert_summary.loc['mean'], yerr=bounded_yerr(pert_summary.loc['mean'], pert_summary.loc['std']), marker='s', label=pert_title_part, capsize=5)
    
    # **FIXED**: Create labels and call the function with the correct arguments
    orig_labels = [f"{v:.3f}" for v in orig_summary.loc['mean']]
    pert_labels = [f"{v:.3f}" for v in pert_summary.loc['mean']]
    add_smart_value_labels(
        x_positions=x_pos,
        values1=orig_summary.loc['mean'],
        values2=pert_summary.loc['mean'],
        labels1_text=orig_labels,
        labels2_text=pert_labels,
        color1='C0',
        color2='C1'
    )


    comparison_title = f"Performance Comparison: Original vs. {pert_title_part}"
    plt.suptitle(comparison_title)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x_pos, metrics, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.figtext(0.02, 0.02, orig_subtitle, ha="left", va="bottom", wrap=True)
    plt.figtext(0.98, 0.02, pert_subtitle, ha="right", va="bottom", wrap=True)
    
    # Use subplots_adjust to make space for the new footer
    plt.subplots_adjust(bottom=0.25)
    plot_save_path = artefact_path(experiment=f"{dataset_base}_vs_{perturbation_tag}", art_type="comparison", filename="comparison_summary.pdf")
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison summary plot to: {plot_save_path}")

    # --- Delta plot ---
    plt.figure()
    for m in metrics:
        merged[f'{m}_delta'] = merged[f"{m}_pert"] - merged[f"{m}_orig"]
    deltas_mean = merged[[f'{m}_delta' for m in metrics]].mean()
    deltas_std = merged[[f'{m}_delta' for m in metrics]].std()
    plt.errorbar(x_pos, deltas_mean, yerr=deltas_std, fmt='-o', capsize=5, color='green')
    
    delta_labels = [f"{v:+.3f}" for v in deltas_mean]
    add_single_series_labels(
        x_positions=x_pos,
        values=deltas_mean,
        labels_text=delta_labels
    )
    
    delta_title = f"Performance Difference: {pert_title_part}"
    plt.suptitle(delta_title)
        
    plt.xlabel('Metrics')
    plt.ylabel('Difference in Score')
    plt.xticks(x_pos, metrics, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.figtext(0.02, 0.02, orig_subtitle, ha="left", va="bottom", wrap=True)
    plt.figtext(0.98, 0.02, pert_subtitle, ha="right", va="bottom", wrap=True)
    
    # Use subplots_adjust to make space for the new footer
    plt.subplots_adjust(bottom=0.25)

    delta_plot_path = artefact_path(experiment=f"{dataset_base}_vs_{perturbation_tag}", art_type="comparison", filename="comparison_delta.pdf")
    plt.savefig(delta_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved delta comparison plot to: {delta_plot_path}")

    print(f"\nAll comparison artifacts saved under: {reports_root() / 'comparisons' / f'{dataset_base}_vs_{pert_family_base}'}")
