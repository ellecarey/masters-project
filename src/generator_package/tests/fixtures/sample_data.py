"""
Sample data and constants for testing the GaussianDataGenerator.
"""

SAMPLE_FEATURE_PARAMS = {
    "test_feature_1": {"mean": 0, "std": 1},
    "test_feature_2": {"mean": 10, "std": 3},
    "test_feature_3": {"mean": -5, "std": 0.5},
}

EXTREME_FEATURE_PARAMS = {
    "high_precision_feature": {"mean": 0.123456789, "std": 0.987654321},
    "large_scale_feature": {"mean": 1e6, "std": 1e3},
    "negative_large_feature": {"mean": -1e6, "std": 1e-3},
}

INVALID_FEATURE_PARAMS = {
    "negative_std": {"mean": 0, "std": -1},
    "zero_std": {"mean": 0, "std": 0},
    "missing_mean": {"std": 1},
    "missing_std": {"mean": 0},
    "wrong_type": "not_a_dict",
    "extra_keys": {"mean": 0, "std": 1, "extra": "value"},
}

# Feature type constants
SAMPLE_FEATURE_TYPES = {
    "test_feature_1": "continuous",
    "test_feature_2": "discrete",
    "test_feature_3": "continuous",
}

INVALID_FEATURE_TYPES = {
    "invalid_categorical": "categorical",
    "invalid_numeric": 123,
    "invalid_none": None,
    "invalid_string": "invalid_type",
}

# Dataset size configurations for different test scenarios
DATASET_CONFIGS = {
    "minimal_valid_input": {"samples": 1, "features": 1},
    "unit_test_standard": {"samples": 50, "features": 3},
    "integration_test_size": {"samples": 500, "features": 10},
    "performance_benchmark": {"samples": 5000, "features": 50},
    "stress_test_maximum": {"samples": 50000, "features": 100},
}

# Perturbation scale constants with descriptive names
PERTURBATION_LEVELS = {
    "noise_free": 0.0,
    "minimal_noise": 0.01,
    "low_noise": 0.05,
    "realistic_noise": 0.1,
    "high_noise": 0.3,
    "extreme_noise": 1.0,
}

# Weight configurations for target variable creation
TARGET_WEIGHTS = {
    "balanced_weights": [1.0, -0.5, 0.3],
    "unbalanced_weights": [2.5, -1.8, 0.7, -0.2, 1.1],
    "single_weight": [1.0],
    "zero_weights": [0.0, 0.0, 0.0],
    "large_magnitude": [100.0, -50.0, 25.0],
    "small_magnitude": [0.001, -0.002, 0.003],
}

# Function types for target creation
FUNCTION_TYPES = ["linear", "polynomial", "logistic"]

# Noise levels for target creation with explicit values
TARGET_NOISE_LEVELS = {
    "no_noise": 0.0,
    "minimal_noise": 0.05,
    "standard_noise": 0.1,
    "high_noise": 0.3,
    "extreme_noise": 0.5,
}

# Random states for reproducibility testing
REPRODUCIBILITY_SEEDS = [42, 123, 999, 2023, 12345]

# Expected value ranges for validation
VALIDATION_RANGES = {
    "default_mean_range": (-5, 5),
    "default_std_range": (0.5, 3),
    "probability_range": (0, 1),
    "binary_values": {0, 1},
}


EDGE_CASE_LIMITS = {
    "minimum_samples": 1,
    "minimum_features": 1,
    "maximum_reasonable_samples": 100000,
    "maximum_reasonable_features": 1000,
}

ERROR_PATTERNS = {
    "negative_samples": r"Number of samples must be positive",
    "negative_features": r"Number of features must be positive",
    "negative_std": r"Standard deviation .* must be positive",
    "missing_parameters": r"Feature .* must have 'mean' and 'std' parameters",
    "invalid_feature_type": r"Invalid feature type",
    "no_data": r"No data generated. Call generate_features\(\) first.",
    "feature_not_found": r"Feature .* not found in data",
    "weight_mismatch": r"Number of weights must match number of features",
    "invalid_function_type": r"function_type must be",
}

# # Performance benchmarks
# PERFORMANCE_THRESHOLDS = {
#     'unit_test_timeout_seconds': 1.0,
#     'integration_test_timeout_seconds': 5.0,
#     'performance_test_timeout_seconds': 30.0,
# }

# # Statistical validation thresholds
# STATISTICAL_TOLERANCES = {
#     'correlation_threshold': 0.95,  # For reproducibility tests
#     'mean_std_tolerance': 0.1,     # For mean/std validation
#     'probability_tolerance': 0.05   # For probability-based tests
# }
