"""
Generator Package - comprehensive data generation toolkit

This package provides tools for generating synthetic datasets with Gaussian distributions,
controlled perturbations, and customisable visualisations.
"""

from .gaussian_data_generator import GaussianDataGenerator
from .plotting_style import apply_custom_plot_style
from . import config
from . import utils

__all__ = ["GuassianDataGenerator", "apply_custom_plot_style", "config", "utils"]


def _initialise_package():
    """Initialise package level configurations."""
    try:
        apply_custom_plot_style()
        print("Generator package initialised.")
    except Exception as e:
        print(f"Warning: could not initialise plotting style: {e}")


_initialise_package()
