import os
from .utils import get_project_paths

PATHS = get_project_paths()
PROJECT_ROOT = PATHS["project_root"]

# --- General Generator Settings ---
N_SAMPLES = 1000
N_INITIAL_FEATURES = 5
RANDOM_STATE = 42

# --- Feature Generation (generate_features) ---
# All parameters must be explicitly defined
FEATURE_PARAMETERS_OVERRIDE = {
    "feature_0": {"mean": 10, "std": 1},
    "feature_1": {"mean": -5, "std": 0.5},
    "feature_2": {"mean": 0, "std": 2},
    "feature_3": {"mean": 3, "std": 1.5},
    "feature_4": {"mean": -2, "std": 0.8},
}

FEATURE_TYPES_OVERRIDE = {
    "feature_0": "discrete",
    "feature_1": "continuous",
    "feature_2": "continuous",
    "feature_3": "discrete",
    "feature_4": "continuous",
}

# --- Feature Perturbation (add_perturbations) ---
PERTURBATION_SETTINGS = {
    "perturbation_type": "gaussian",  # Must be 'gaussian' or 'uniform'
    "features_to_perturb": ["feature_0", "feature_1"],  # Must specify features
    "scale": 0.15,  # Must specify scale
}

# --- Adding New Features (add_features) ---
ADD_FEATURES_SETTINGS = {
    "n_new_features": 2,
    "feature_parameters": {
        "feature_5": {"mean": 5, "std": 1.2},
        "feature_6": {"mean": -3, "std": 0.9},
    },
    "feature_types": {"feature_5": "continuous", "feature_6": "discrete"},
}

# --- Target Variable Creation (create_target_variable) ---
CREATE_TARGET_SETTINGS = {
    "features_to_use": ["feature_0", "feature_1", "feature_2"],
    "weights": [0.5, -0.3, 1.2],
    "noise_level": 0.05,
    "function_type": "polynomial",  # Must be 'linear', 'polynomial', or 'logistic'
}

# --- Visualisation Settings (visualise_features) ---
VISUALISATION_SETTINGS = {
    "features_to_visualise": ["feature_0", "feature_1", "feature_2", "feature_3"],
    "max_features_to_show": 6,
    "n_bins": 25,
    "save_to_dir": None,  # Specify directory path or None
}

# --- Output Settings ---
OUTPUT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
FIGURES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
