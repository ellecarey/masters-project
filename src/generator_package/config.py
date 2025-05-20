# --- General Generator Settings ---
N_SAMPLES = 1000
N_INITIAL_FEATURES = 5  # Initial number of features
RANDOM_STATE = 42  # Seed for all random operations

# --- Feature Generation (generate_features) ---
# To define specific means, stds, and types for initial features.
# If None, the generator will use its default random generation.
FEATURE_PARAMETERS_OVERRIDE = None
# Example:
# FEATURE_PARAMETERS_OVERRIDE = {
#     'feature_0': {'mean': 10, 'std': 1},
#     'feature_1': {'mean': -5, 'std': 0.5}
# }
FEATURE_TYPES_OVERRIDE = None
# Example:
# FEATURE_TYPES_OVERRIDE = {
#     'feature_0': 'discrete',
#     'feature_1': 'continuous'
# }

# --- Feature Perturbation (add_perturbations) ---
# Set to None if no perturbation is needed
PERTURBATION_SETTINGS = {
    "perturbation_type": "gaussian",  # 'gaussian' or 'uniform'
    "features_to_perturb": [
        "feature_0",
        "feature_1",
    ],  # List of feature names, or None for all
    "scale": 0.15,  # Scale of the perturbation
}
# PERTURBATION_SETTINGS = None # Example: To skip perturbations

# --- Adding New Features (add_features) ---
# Set to None if no new features are to be added after initial generation
ADD_FEATURES_SETTINGS = {
    "n_new_features": 2,
    "feature_parameters": None,  # Optional: Define parameters for new features
    "feature_types": None,  # Optional: Define types for new features
}
# ADD_FEATURES_SETTINGS = None # Example: To skip adding features

# --- Target Variable Creation (create_target_variable) ---
# Set to None if no target variable is needed
CREATE_TARGET_SETTINGS = {
    "features_to_use": [
        "feature_0",
        "feature_1",
        "feature_2",
    ],  # List of features, or None for all
    "weights": [0.5, -0.3, 1.2],  # List of weights, or None for random weights
    "noise_level": 0.05,
    "function_type": "polynomial",  # 'linear', 'polynomial', or 'logistic'
}
# CREATE_TARGET_SETTINGS = None # Example: To skip target creation

# --- Visualization Settings (visusalise_features) ---
VISUALIZATION_SETTINGS = {
    "features_to_visualise": None,  # List of specific features, or None for a selection
    "max_features_to_show": 6,  # Max features to show if 'features_to_visualise' is None
    "n_bins": 25,
}

# --- Output Settings ---
OUTPUT_DATA_PATH = "data/dataset.csv"
FIGURES_OUTPUT_DIR = "reports/figures/"
