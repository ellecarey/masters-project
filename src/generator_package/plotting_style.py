import matplotlib.pyplot as plt
import warnings


def apply_custom_plot_style():
    """
    Applies a consistent, publication-quality style to Matplotlib plots.
    Includes settings for LaTeX rendering if available.
    """

    # Reset the parameters to their defaults first to ensure a clean slate
    plt.rcParams.update(plt.rcParamsDefault)

    try:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["pgf.texsystem"] = "pdflatex"  # Specify the TeX system
        # For LaTeX, a serif font like Computer Modern is standard and usually works best
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [
            "Computer Modern Roman"
        ]  # Or 'Computer Modern Serif', 'CMU Serif' etc.
        # pgf.rcfonts = False allows Matplotlib to handle fonts, which is often simpler
        # unless you have very specific needs for pgf to control all fonts directly.
        plt.rcParams["pgf.rcfonts"] = False
        print("Matplotlib configured to use LaTeX for text rendering.")
    except Exception as e:
        print(
            f"Note: LaTeX rendering for Matplotlib could not be enabled. Using default text rendering. Error: {e}"
        )
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = (
            "sans-serif"  # Fallback to sans-serif if LaTeX fails
        )

    # Define common parameters
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "figure.titlesize": 20,
        "axes.titlesize": 16,  # For individual subplot titles
        "axes.labelsize": 14,  # For x and y labels
        "xtick.labelsize": 14,  # For x-axis tick labels
        "ytick.labelsize": 14,  # For y-axis tick labels
        "figure.figsize": [
            46.82 * 0.5 ** (0.5 * 6),
            33.11 * 0.5 ** (0.5 * 6),
        ],  # Specific figure size
        "figure.dpi": 300,
    }
    plt.rcParams.update(params)

    # Apply a base style sheet
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        print(
            "Warning: 'seaborn-v0_8-whitegrid' style not found. Consider updating Matplotlib or using an available style."
        )
        # Fallback to a basic grid if the style is not found
        plt.rcParams["axes.grid"] = True

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    print("Custom Matplotlib plot style applied.")
