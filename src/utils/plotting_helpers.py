import numpy as np
import textwrap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from adjustText import adjust_text


def generate_subtitle_from_config(config: dict) -> str:
    """
    Generates a detailed, human-readable subtitle from a configuration dictionary.
    """
    try:
        ds_settings = config.get("dataset_settings", {})
        n_samples = ds_settings.get("n_samples", "N/A")
        
        class_config = config.get("create_feature_based_signal_noise_classification", {})
        feature_types = class_config.get("feature_types", {})
        
        total_features = len(feature_types)
        continuous_count = sum(1 for ft in feature_types.values() if ft == "continuous")
        discrete_count = sum(1 for ft in feature_types.values() if ft == "discrete")
        
        feature_desc = f"{total_features} Features ({continuous_count} Cont, {discrete_count} Disc)"
        
        # Enhanced perturbation handling with better error recovery
        pert_settings = config.get("perturbation_settings")
        pert_desc = _process_perturbation_settings(pert_settings)
        
        signal_features = class_config.get("signal_features", {})
        noise_features = class_config.get("noise_features", {})
        
        separations = []
        for f_name, s_params in signal_features.items():
            if f_name in noise_features:
                n_params = noise_features[f_name]
                mean_diff = abs(s_params.get('mean', 0) - n_params.get('mean', 0))
                s_std = s_params.get('std', 1)
                n_std = n_params.get('std', 1)
                if (s_std**2 + n_std**2) > 0:
                    d = mean_diff / ((s_std**2 + n_std**2)**0.5)
                    separations.append(d)

        overall_separation = (sum(d**2 for d in separations))**0.5 if separations else 0.0
        separation_desc = f"Std. Separation: {overall_separation:.2f}"
        
        subtitle = (
            f"Dataset: {n_samples:,} Samples, {feature_desc}\n"
            f"{pert_desc} | {separation_desc}"
        )
        
        return subtitle
        
    except Exception as e:
        # More specific error reporting for debugging
        return f"Configuration parsing error: {str(e)[:50]}..."

def _process_perturbation_settings(pert_settings) -> str:
    """
    Process perturbation settings with robust error handling.
    """
    if not pert_settings or not isinstance(pert_settings, list) or not pert_settings:
        return "No Perturbations"
    
    pert_descs = []
    
    for i, p in enumerate(pert_settings):
        try:
            if not isinstance(p, dict):
                pert_descs.append(f"Invalid perturbation {i+1}")
                continue
                
            pert_type = p.get('type', 'individual')
            class_label = p.get('class_label')
            
            # Validate class_label
            if class_label not in [0, 1]:
                pert_descs.append(f"Invalid class label in perturbation {i+1}")
                continue
                
            class_name = 'Noise' if class_label == 0 else 'Signal'
            
            if pert_type == 'correlated':
                desc = _process_correlated_perturbation(p, class_name)
            else:
                desc = _process_individual_perturbation(p, class_name)
                
            if desc:
                pert_descs.append(desc)
                
        except Exception as e:
            pert_descs.append(f"Error in perturbation {i+1}: {str(e)[:30]}")
    
    return f"Perturbation: {'; '.join(pert_descs)}" if pert_descs else "No valid perturbations"

def _process_correlated_perturbation(p: dict, class_name: str) -> str:
    """Process correlated perturbation with error handling."""
    try:
        features = p.get('features', [])
        if not features:
            return f"Correlated {class_name} (no features specified)"
            
        description = p.get('description', '')
        
        # Create feature string
        if len(features) <= 3:
            feature_str = ', '.join([f.replace('feature_', 'F') for f in features])
        else:
            feature_str = f"{len(features)} features"
        
        # Handle different perturbation parameters
        if 'scale_factor' in p:
            scale_val = p.get('scale_factor', 'N/A')
            base_desc = f"Corr {feature_str} ({class_name}) {scale_val}x"
        elif 'sigma_shift' in p:
            shift_val = p.get('sigma_shift', 'N/A')
            try:
                shift_val_formatted = f"{float(shift_val):+.1f}σ"
            except (ValueError, TypeError):
                shift_val_formatted = f"{shift_val}σ"
            base_desc = f"Corr {feature_str} ({class_name}) {shift_val_formatted}"
        else:
            base_desc = f"Corr {feature_str} ({class_name})"
        
        if description:
            return f"{base_desc} [{description}]"
        else:
            return base_desc
            
    except Exception as e:
        return f"Correlated {class_name} (parsing error)"

def _process_individual_perturbation(p: dict, class_name: str) -> str:
    """Process individual perturbation with error handling."""
    try:
        feature = p.get('feature', 'N/A')
        
        if 'scale_factor' in p:
            scale_val = p.get('scale_factor', 'N/A')
            return f"{feature} ({class_name}) scaled by {scale_val}x"
        elif 'sigma_shift' in p:
            shift_val = p.get('sigma_shift', 'N/A')
            try:
                shift_val_formatted = f"{float(shift_val):+.1f}σ"
            except (ValueError, TypeError):
                shift_val_formatted = f"{shift_val}σ"
            return f"{feature} ({class_name}) by {shift_val_formatted}"
        elif 'additive_noise' in p:
            noise_val = p.get('additive_noise', 'N/A')
            return f"{feature} ({class_name}) +noise {noise_val}"
        elif 'multiplicative_factor' in p:
            mult_val = p.get('multiplicative_factor', 'N/A')
            return f"{feature} ({class_name}) ×{mult_val}"
        else:
            return f"{feature} ({class_name}) perturbation"
            
    except Exception as e:
        return f"{class_name} perturbation (parsing error)"


def bounded_yerr(mean, spread, lo=0.0, hi=1.0):
    """
    Return asymmetric y-err that never crosses [lo, hi] bounds.
    """
    mean = np.asarray(mean)
    spread = np.asarray(spread)
    upper = np.minimum(mean + spread, hi) - mean
    lower = mean - np.maximum(mean - spread, lo)
    return np.vstack([lower, upper])


def calculate_adaptive_ylimits(data_mean, data_std, padding_factor=0.2, bound_to_0_1=True):
    """
    Calculates adaptive y-axis limits based on data mean and std dev.
    """
    min_val = np.min(data_mean - data_std)
    max_val = np.max(data_mean + data_std)
    data_range = max_val - min_val
    padding = data_range * padding_factor
    y_min = min_val - padding
    y_max = max_val + padding

    if np.isclose(data_range, 0):
        y_min = min_val - 0.1
        y_max = max_val + 0.1

    if bound_to_0_1:
        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)

    return y_min, y_max


def add_smart_value_labels(ax, x_positions, values1, stds1, values2, stds2,
                           color1='blue', color2='red', fontsize=9, fontweight='bold'):
    """
    MODIFIED: Adds non-overlapping value labels using the adjustText library.
    This version uses bold font and more aggressive settings to ensure labels
    are clearly separated and readable even in crowded plot areas.
    """
    labels1_text = [f"{v:.3f} \u00B1 {s:.3f}" for v, s in zip(values1, stds1)]
    labels2_text = [f"{v:.3f} \u00B1 {s:.3f}" for v, s in zip(values2, stds2)]

    texts = []
    # Create text objects for the first series
    for x, y, label in zip(x_positions, values1, labels1_text):
        texts.append(ax.text(x, y, label, color=color1, fontsize=fontsize, fontweight=fontweight, ha='center'))

    # Create text objects for the second series
    for x, y, label in zip(x_positions, values2, labels2_text):
        texts.append(ax.text(x, y, label, color=color2, fontsize=fontsize, fontweight=fontweight, ha='center'))

    # Use adjust_text to automatically position all labels to avoid any overlap
    adjust_text(
        texts,
        ax=ax,
        # Draw clear arrows from labels to their data points
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.75, alpha=0.8),
        # Use stronger forces to push labels apart from each other and from data points
        force_text=(0.8, 1.2),
        force_points=(0.5, 0.5)
    )


def add_single_series_labels(ax, x_positions, values, labels_text, color='green',
                             fontsize=None, fontweight='normal'):
    """
    Add labels for a single data series, using adjustText to prevent overlaps.
    """
    texts = []
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.025

    for x, y, label in zip(x_positions, values, labels_text):
        texts.append(ax.text(x, y + offset, label, color="black", fontsize=fontsize, fontweight=fontweight, ha='center', va='bottom'))

    # Use adjust_text to automatically position all labels
    adjust_text(
        texts,
        ax=ax,
        force_text=(0.5, 0.5), # Repel labels from each other
        force_points=(0.2, 0.2),# Weakly repel from data points (offset does most of the work)
    )


def format_plot_title(main_title: str, experiment_name: str, main_width: int = 60, sub_width: int = 80):
    """
    Formats a plot title by wrapping the main title and a long experiment name.
    """
    wrapped_main = textwrap.fill(main_title, width=main_width)
    wrapped_sub = textwrap.fill(experiment_name, width=sub_width)
    return f"{wrapped_main}\n({wrapped_sub})"


def apply_decimal_formatters(ax, precision=3):
    """
    Apply a specific decimal precision to the y-axis tick labels.
    """
    formatter = mticker.FormatStrFormatter(f'%.{precision}f')
    ax.yaxis.set_major_formatter(formatter)

