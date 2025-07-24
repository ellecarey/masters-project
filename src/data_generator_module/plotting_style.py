import matplotlib.pyplot as plt
import warnings

def apply_custom_plot_style():
    """
    Applies a consistent, publication-quality style to Matplotlib plots
    using a reliable, non-LaTeX text rendering engine.
    """
    # Reset to defaults to avoid style conflicts
    plt.rcParams.update(plt.rcParamsDefault)

    # --- Font and Text Settings (No LaTeX) ---
    # Explicitly disable TeX rendering
    plt.rcParams["text.usetex"] = False
    # Use a clean, professional sans-serif font available by default
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Arial",
        "Helvetica",
        "Avant Garde",
        "sans-serif",
    ]
    
    # Ensure fonts are embedded in PDF/PS files for portability
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # --- General Plot Styling (Preserved from original) ---
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "figure.titlesize": 20,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": "medium",
        "figure.figsize": [12, 7],
        "figure.dpi": 300,
        "figure.subplot.wspace": 0.3,
        "figure.subplot.hspace": 0.3,
    }
    plt.rcParams.update(params)

    # Use a professional-looking plot style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        print("Warning: 'seaborn-v0_8-whitegrid' style not found. Using basic grid.")
        plt.rcParams["axes.grid"] = True

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    print("Custom Matplotlib plot style applied (LaTeX rendering disabled).")

