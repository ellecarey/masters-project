# src/analysis_module/comparison.py

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
from src.utils.report_paths import artefact_path, reports_root, experiment_family_path
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

    safe_perturbation_tag = perturbation_tag.lstrip('_')

    # --- 1. & 2. Data Loading ---
    dataset_base, model_name, _, _ = parse_optimal_config_name(orig_config_path)
    dataset_base = dataset_base.rstrip('_')
    models_dir = project_root / "models"
    orig_pattern = metrics_filename(dataset_base, model_name, seed="*", perturbation_tag=None, optimized=False)
    orig_metric_files = list(models_dir.glob(orig_pattern.replace('*', '[0-9]*')))

    if not orig_metric_files:
        print(f"Error: No metric files found for original family '{dataset_base}'.")
        return

    all_seeds = sorted([int(re.search(r'_seed(\d+)', f.stem).group(1)) for f in orig_metric_files])
    seeds = [s for s in all_seeds if s != TRAINING_SEED]
    print(f"Discovered and using {len(seeds)} evaluation seeds for comparison: {seeds}")

    orig_metrics_files = [models_dir / metrics_filename(dataset_base, model_name, seed=s, optimized=False) for s in seeds]
    pert_metrics_files = [models_dir / metrics_filename(dataset_base, model_name, seed=s, perturbation_tag=safe_perturbation_tag, optimized=False) for s in seeds]

    orig_metrics_data = [json.loads(f.read_text()) for f in orig_metrics_files if f.exists()]
    pert_metrics_data = [json.loads(f.read_text()) for f in pert_metrics_files if f.exists()]

    if not orig_metrics_data or not pert_metrics_data:
        print("\nError: Metric files are missing. Cannot create comparison.")
        if not orig_metrics_data:
            print("Could not find metric files for the ORIGINAL family.")
        if not pert_metrics_data:
            print("Could not find metric files for the PERTURBED family.")
        return

    orig_df = pd.DataFrame(orig_metrics_data)
    orig_df['seed'] = seeds
    pert_df = pd.DataFrame(pert_metrics_data)
    pert_df['seed'] = seeds

    ## --- 3. Generate Dynamic Title ---
    pert_family_base = f"{dataset_base}_{safe_perturbation_tag}"
    pert_data_config_path = project_root / "configs" / "data_generation" / f"{pert_family_base}_seed0_config.yml"
    
    pert_title_part = "Perturbed"
    if pert_data_config_path.exists():
        pert_config = data_utils.load_yaml_config(str(pert_data_config_path))
        pert_settings = pert_config.get("perturbation_settings")
        if pert_settings and isinstance(pert_settings, list) and pert_settings:
            pert_descs = []
            for p in pert_settings:
                pert_type = p.get('type', 'individual')
                class_label = 'Noise' if p.get('class_label') == 0 else 'Signal'
                if pert_type == 'correlated':
                    features = p.get('features', [])
                    description = p.get('description', '')
                    if len(features) <= 2:
                        feature_str = '+'.join([f.replace('feature_', 'F') for f in features])
                    else:
                        feature_str = f"{len(features)} Features"
                    
                    if 'scale_factor' in p:
                        scale_val = p.get('scale_factor', 'N/A')
                        base_desc = f"Corr {feature_str} ({class_label}) {scale_val}x"
                    elif 'sigma_shift' in p:
                        shift_val = p.get('sigma_shift', 'N/A')
                        base_desc = f"Corr {feature_str} ({class_label}) {shift_val:+.1f}σ"
                    
                    if description:
                        pert_descs.append(f"{base_desc} [{description}]")
                    else:
                        pert_descs.append(base_desc)
                        
                else:
                    # Handle individual perturbations (existing code)
                    feature = p.get('feature', 'N/A')
                    if 'scale_factor' in p:
                        scale_val = p.get('scale_factor', 'N/A')
                        pert_descs.append(f"{feature} ({class_label}) scaled by {scale_val}x")
                    elif 'sigma_shift' in p:
                        shift_val = p.get('sigma_shift', 'N/A')
                        pert_descs.append(f"{feature} ({class_label}) by {shift_val:+.1f}σ")
                        
            pert_title_part = "; ".join(pert_descs)
    
    orig_data_config_path = project_root / "configs" / "data_generation" / f"{dataset_base}_seed0_config.yml"
    orig_subtitle = "Original:\n" + (generate_subtitle_from_config(data_utils.load_yaml_config(orig_data_config_path)) if orig_data_config_path.exists() else "Details unavailable")
    pert_subtitle = "Perturbed:\n" + (generate_subtitle_from_config(data_utils.load_yaml_config(pert_data_config_path)) if pert_data_config_path.exists() else "Details unavailable")

    # --- 4. Plotting ---
    merged = pd.merge(orig_df, pert_df, on='seed', suffixes=('_orig', '_pert'))
    metrics = [col for col in orig_df.columns if col not in ['seed', 'Test Loss (BCE)']]
    x_pos = np.arange(len(metrics))

    # --- Main comparison plot ---
    fig, ax = plt.subplots()
    orig_summary = orig_df[metrics].agg(['mean', 'std'])
    pert_summary = pert_df[metrics].agg(['mean', 'std'])
    
    all_means = pd.concat([orig_summary.loc['mean'], pert_summary.loc['mean']])
    all_stds = pd.concat([orig_summary.loc['std'], pert_summary.loc['std']])
    y_min, y_max = calculate_adaptive_ylimits(all_means, all_stds, bound_to_0_1=True, padding_factor=0.1)
    ax.set_ylim(y_min, y_max)


    ax.errorbar(x_pos, orig_summary.loc['mean'], yerr=bounded_yerr(orig_summary.loc['mean'], orig_summary.loc['std']), marker='o', label='Original', capsize=10, color='darkblue', linestyle='-')
    _, _, bars_pert = ax.errorbar(x_pos, pert_summary.loc['mean'], yerr=bounded_yerr(pert_summary.loc['mean'], pert_summary.loc['std']), marker='s', label=pert_title_part, capsize=5, color='firebrick', linestyle='-')
    
    [bar.set_linestyle('--') for bar in bars_pert]

    comparison_title = f"Performance Comparison: Original vs. {pert_title_part}"
    fig.suptitle(comparison_title, y=0.98, fontsize=16)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    fig.text(0.02, 0.02, orig_subtitle, ha="left", va="bottom", wrap=True)
    fig.text(0.98, 0.02, pert_subtitle, ha="right", va="bottom", wrap=True)
    fig.subplots_adjust(bottom=0.25, top=0.9)
    
    # Save comparison summary plot
    plot_save_path = experiment_family_path(
        full_experiment_name=dataset_base,
        art_type="figure", 
        subfolder="comparison_summary",
        filename=f"{dataset_base}_{safe_perturbation_tag}_comparison_summary.pdf"
    )
    fig.savefig(plot_save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison summary plot to: {plot_save_path}")

    # --- Delta plot ---
    fig_delta, ax_delta = plt.subplots()
    for m in metrics:
        merged[f'{m}_delta'] = merged[f"{m}_pert"] - merged[f"{m}_orig"]

    deltas_mean = merged[[f'{m}_delta' for m in metrics]].mean()
    deltas_std = merged[[f'{m}_delta' for m in metrics]].std()
    
    y_min_delta, y_max_delta = calculate_adaptive_ylimits(deltas_mean, deltas_std, padding_factor=0.4, bound_to_0_1=False)
    ax_delta.set_ylim(y_min_delta, y_max_delta)

    ax_delta.errorbar(x_pos, deltas_mean, yerr=deltas_std, fmt='-o', capsize=5, color='green')
    delta_labels = [f"{v:+.3f} \u00B1 {s:.4f}" for v, s in zip(deltas_mean, deltas_std)]

    # add_single_series_labels(
    #     ax=ax_delta,
    #     x_positions=x_pos,
    #     values=deltas_mean,
    #     labels_text=delta_labels,
    #     color='darkgreen'
    # )

    delta_title = f"Performance Difference (Perturbed - Original)"
    fig_delta.suptitle(delta_title, y=0.98)
    ax_delta.set_xlabel('Metrics')
    ax_delta.set_ylabel('Difference in Score')
    ax_delta.set_xticks(x_pos)
    ax_delta.set_xticklabels(metrics, rotation=45, ha='right')
    ax_delta.grid(True, linestyle='--', alpha=0.7)

    fig_delta.text(0.02, 0.02, orig_subtitle, ha="left", va="bottom", wrap=True)
    fig_delta.text(0.98, 0.02, pert_subtitle, ha="right", va="bottom", wrap=True)
    fig_delta.subplots_adjust(bottom=0.25, top=0.9)
    
    # Save delta comparison plot  
    delta_plot_path = experiment_family_path(
        full_experiment_name=dataset_base,
        art_type="figure", 
        subfolder="comparison_summary",
        filename=f"{dataset_base}_{safe_perturbation_tag}_delta_summary.pdf"
    )

    fig_delta.savefig(delta_plot_path, bbox_inches='tight')
    plt.close(fig_delta)
    print(f"Saved delta comparison plot to: {delta_plot_path}")