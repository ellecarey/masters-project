import matplotlib.pyplot as plt
import warnings


def apply_custom_plot_style(use_latex: bool = True):
    """
    Applies a consistent, publication-quality style to Matplotlib plots.
    Includes settings for LaTeX rendering if available and requested.

    Parameters:
    -----------
    use_latex (bool): Attempt to use LaTeX for text rendering. Defaults to True.
    """

    plt.rcParams.update(plt.rcParamsDefault)

    can_use_latex = False
    if use_latex:
        try:
            from matplotlib.texmanager import TexManager

            if TexManager().check_availability():  # Basic check for LaTeX
                plt.rcParams["text.usetex"] = True
                plt.rcParams["pgf.texsystem"] = "pdflatex"
                plt.rcParams["font.family"] = "serif"
                plt.rcParams["font.serif"] = ["Computer Modern Roman"]
                plt.rcParams["pgf.rcfonts"] = False
                can_use_latex = True
                print("Matplotlib configured to use LaTeX for text rendering.")
            else:
                print(
                    "Warning: LaTeX installation not found or not fully functional. Falling back to default text rendering."
                )
        except Exception as e:
            print(
                f"Warning: Could not enable LaTeX for Matplotlib. Error: {e}. Falling back to default text rendering."
            )

    if not can_use_latex:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "sans-serif"

    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "figure.titlesize": 20,  # For fig.suptitle()
        "axes.titlesize": 16,  # For ax.set_title()
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": "medium",  # Added for better legend readability
        "figure.figsize": [12, 7],  # Was [10, 7]
        "figure.dpi": 300,
        # Add some default inter-subplot spacing; tight_layout can still override/refine this
        "figure.subplot.wspace": 0.3,  # Width space between subplots
        "figure.subplot.hspace": 0.3,  # Height space between subplots
    }
    plt.rcParams.update(params)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        print("Warning: 'seaborn-v0_8-whitegrid' style not found. Using basic grid.")
        plt.rcParams["axes.grid"] = True

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    print("Custom Matplotlib plot style applied.")
