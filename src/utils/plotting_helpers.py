import numpy as np
import textwrap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def bounded_yerr(mean, spread, lo=0.0, hi=1.0):
    """
    Return asymmetric y-err that never crosses [lo, hi] bounds.
    
    Parameters:
    -----------
    mean : array-like
        Mean values
    spread : array-like  
        Standard deviation or error spread
    lo : float
        Lower bound (default: 0.0)
    hi : float
        Upper bound (default: 1.0)
        
    Returns:
    --------
    np.ndarray
        2D array with lower and upper error bounds
    """
    mean = np.asarray(mean)
    spread = np.asarray(spread)
    
    upper = np.minimum(mean + spread, hi) - mean
    lower = mean - np.maximum(mean - spread, lo)
    return np.vstack([lower, upper])


def calculate_adaptive_ylimits(mean_values, std_values, padding_factor=0.1, annotation_padding=0.02):
    """
    Calculate visually appealing y-limits based on data range.
    
    Parameters:
    -----------
    annotation_padding : float
        Extra padding to account for annotation text
    """
    mean_values = np.asarray(mean_values)
    std_values = np.asarray(std_values)
    
    min_val = np.max([np.min(mean_values - std_values), 0.0])
    max_val = np.min([np.max(mean_values + std_values), 1.0])
    
    data_range = max_val - min_val
    
    # If metrics are tightly clustered, zoom in
    if data_range < 0.1:
        center = (max_val + min_val) / 2
        zoom_range = max(0.1, data_range * 2)  # At least 0.1 range
        y_min = max(0.0, center - zoom_range/2)
        y_max = min(1.05, center + zoom_range/2)
    else:
        padding = data_range * padding_factor
        y_min = max(0.0, min_val - padding)
        y_max = min(1.05, max_val + padding)
    
    # Add extra space for annotations
    final_range = y_max - y_min
    annotation_space = final_range * annotation_padding
    y_min = max(0.0, y_min - annotation_space)
    y_max = min(1.05, y_max + annotation_space)
    
    return y_min, y_max

def add_smart_value_labels(x_positions, values1, values2, labels1_text, labels2_text,
                           color1='blue', color2='red', fontsize=None, fontweight='normal'):
    """
    Add smart value labels that position themselves based on which value is higher.
    """
    # If no fontsize is given, use the global default for 'axes.labelsize'
    if fontsize is None:
        fontsize = plt.rcParams.get('axes.labelsize', 10)

    for i, (x, val1, val2, text1, text2) in enumerate(zip(x_positions, values1, values2,
                                                         labels1_text, labels2_text)):
        if val1 > val2:
            # First series above, second series below
            plt.text(x, val1, text1,
                     ha='center', va='bottom', fontsize=fontsize,
                     color=color1, fontweight=fontweight)
            plt.text(x, val2, text2,
                     ha='center', va='top', fontsize=fontsize,
                     color=color2, fontweight=fontweight)
        else:
            # Second series above, first series below
            plt.text(x, val2, text2,
                     ha='center', va='bottom', fontsize=fontsize,
                     color=color2, fontweight=fontweight)
            plt.text(x, val1, text1,
                     ha='center', va='top', fontsize=fontsize,
                     color=color1, fontweight=fontweight)


def add_single_series_labels(x_positions, values, labels_text, color='green',
                             fontsize=None, fontweight='normal', offset_above=10, offset_below=-15):
    """
    Add labels for a single data series with smart positioning.
    """
    # If no fontsize is given, use a slightly smaller size than the global default
    if fontsize is None:
        fontsize = plt.rcParams.get('axes.labelsize', 10) * 0.9

    for i, (x, val, text) in enumerate(zip(x_positions, values, labels_text)):
        offset = offset_above if val >= 0 else offset_below
        plt.annotate(text,
                     (x, val),
                     textcoords="offset points",
                     xytext=(0, offset),
                     ha='center', fontsize=fontsize,
                     color=color, fontweight=fontweight)

def format_plot_title(main_title: str, experiment_name: str, main_width: int = 60, sub_width: int = 80):
    """
    Formats a plot title by wrapping the main title and a long experiment name.

    Args:
        main_title (str): The primary title text.
        experiment_name (str): The potentially long experiment name to be wrapped.
        main_width (int): The wrapping width for the main title.
        sub_width (int): The wrapping width for the experiment name subtitle.

    Returns:
        str: A formatted, multi-line title string.
    """
    wrapped_main = textwrap.fill(main_title, width=main_width)
    wrapped_sub = textwrap.fill(experiment_name, width=sub_width)
    return f"{wrapped_main}\n({wrapped_sub})"

def apply_decimal_formatters(ax, precision=3):
    """
    Apply a specific decimal precision to the y-axis tick labels.
    
    Args:
        ax: The Matplotlib axes object to format.
        precision (int): The number of decimal places to show.
    """
    formatter = mticker.FormatStrFormatter(f'%.{precision}f')
    ax.yaxis.set_major_formatter(formatter)