import yaml
import os
import numpy as np

# --- Central Configuration ---
OUTPUT_DIR = "configs/perturbation/disc5_sep1p8/combined-0p35"
FEATURES = [f"feature_{i}" for i in range(5)] # update for how many features you have
CLASS_LABEL_TO_PERTURB = 0
CLASS_NAME_IN_FILENAME = 'n'

# Only generate files if the new separability is below this percentage of the original.
# Example: 0.95 means the new separability must be < 95% of the original (a 5% reduction).
SEPARATION_REDUCTION_THRESHOLD = 0.35

# This dictionary holds the data needed to determine shift direction. (need to update for the dataset you are generating perturbations for)
FEATURE_MEANS = {
    'feature_0': {'noise': 2.0, 'signal': 3.0},
    'feature_1': {'noise': 2.5, 'signal': 1.0},
    'feature_2': {'noise': 2.0, 'signal': 3.5},
    'feature_3': {'noise': 3.0, 'signal': -2.0},
    'feature_4': {'noise': 1.0, 'signal': 2.5},
}

FEATURE_STD_DEVS = {
    'feature_0': {'noise': 2.8, 'signal': 2.0},
    'feature_1': {'noise': 2.5, 'signal': 3.5},
    'feature_2': {'noise': 3.0, 'signal': 1.5},
    'feature_3': {'noise': 2.0, 'signal': 2.5},
    'feature_4': {'noise': 1.8, 'signal': 1.0},
}

# --- Perturbation strengths ---
SCALE_FACTORS = [1.25, 1.5, -1.25, -1.5]

# Use these for features where noise_mean < signal_mean (shift RIGHT)
POSITIVE_SIGMA_SHIFTS = [0.25, 0.5]

# Use these for features where noise_mean > signal_mean (shift LEFT)
NEGATIVE_SIGMA_SHIFTS = [-0.25, -0.5]

# Correlation matrices (key = number of features)
CORRELATION_MATRICES = {
    2: [[1.0, 0.8], [0.8, 1.0]],
    3: [[1.0, 0.7, 0.5], [0.7, 1.0, 0.6], [0.5, 0.6, 1.0]],
    4: [[1.0, 0.7, 0.5, 0.4], [0.7, 1.0, 0.6, 0.3], [0.5, 0.6, 1.0, 0.2], [0.4, 0.3, 0.2, 1.0]],
    5: [[1.0, 0.7, 0.5, 0.4, 0.3], [0.7, 1.0, 0.6, 0.3, 0.2], [0.5, 0.6, 1.0, 0.2, 0.1], [0.4, 0.3, 0.2, 1.0, 0.05], [0.3, 0.2, 0.1, 0.05, 1.0]]
}

# --- Helper Functions ---
def format_val(value):
    """Converts a number to a string for a filename (e.g., -1.5 -> '-1p5')."""
    return str(value).replace('.', 'p')

def write_yaml_file(config_data, filename):
    """Saves the configuration dictionary to a YAML file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)
    print(f"Generated: {filepath}")

# --- Separability Calculation Logic ---
def calculate_separability(noise_means, noise_vars):
    d_squared_sum = 0
    for feature in FEATURES:
        mu_s = FEATURE_MEANS[feature]['signal']
        mu_n = noise_means[feature]
        var_s = FEATURE_STD_DEVS[feature]['signal']**2
        var_n = noise_vars[feature]
        
        denominator = np.sqrt(var_s + var_n)
        if denominator == 0: continue
        d_ic = abs(mu_s - mu_n) / denominator
        d_squared_sum += d_ic**2
        
    return np.sqrt(d_squared_sum)

# Pre-calculate original values for all scripts to use
ORIGINAL_NOISE_MEANS = {f: FEATURE_MEANS[f]['noise'] for f in FEATURES}
ORIGINAL_NOISE_VARS = {f: FEATURE_STD_DEVS[f]['noise']**2 for f in FEATURES}
ORIGINAL_SEPARATION = calculate_separability(ORIGINAL_NOISE_MEANS, ORIGINAL_NOISE_VARS)