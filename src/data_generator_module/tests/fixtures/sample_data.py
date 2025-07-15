"""
Updated sample data for feature-based signal vs noise classification testing.
"""

"""
Updated sample data for feature-based signal vs noise classification testing.
"""

# Feature parameters for testing
SAMPLE_FEATURE_PARAMS = {
    "feature_0": {"mean": 0.0, "std": 1.0},
    "feature_1": {"mean": 10.0, "std": 2.0},
    "feature_2": {"mean": -5.0, "std": 1.5},
}

# Extreme feature parameters for edge case testing
EXTREME_FEATURE_PARAMS = {
    "feature_0": {"mean": 1000.0, "std": 500.0},
    "feature_1": {"mean": -1000.0, "std": 0.001},
    "feature_2": {"mean": 0.0, "std": 1000.0},
}

# Invalid feature parameters for error testing
INVALID_FEATURE_PARAMS = {
    "missing_std": {"mean": 0.0},
    "negative_std": {"mean": 0.0, "std": -1.0},
    "zero_std": {"mean": 0.0, "std": 0.0},
    "non_numeric_mean": {"mean": "invalid", "std": 1.0},
    "non_numeric_std": {"mean": 0.0, "std": "invalid"},
}

# Feature types for testing
SAMPLE_FEATURE_TYPES = {
    "feature_0": "continuous",
    "feature_1": "discrete",
    "feature_2": "continuous",
}

# Invalid feature types for error testing
INVALID_FEATURE_TYPES = {
    "feature_0": "invalid_type",
    "feature_1": "categorical",
    "feature_2": 123,
}

# Target weights for testing
TARGET_WEIGHTS = {
    "balanced": [0.5, -0.5, 0.3],
    "positive": [0.8, 0.2, 0.5],
    "negative": [-0.3, -0.7, -0.1],
    "mixed": [1.0, -1.0, 0.5, -0.2],
}

# Feature noise levels for testing
FEATURE_NOISE_LEVELS = {
    "low": 0.1,
    "medium": 0.5,
    "high": 1.0,
}

# Function types for testing
FUNCTION_TYPES = ["linear", "polynomial", "logistic"]

# Target noise levels for testing
TARGET_NOISE_LEVELS = {
    "none": 0.0,
    "low": 0.1,
    "medium": 0.3,
    "high": 0.5,
}

# Signal/noise parameters for testing
SIGNAL_DISTRIBUTION_PARAMS = {"mean": 2.0, "std": 0.8}
NOISE_DISTRIBUTION_PARAMS = {"mean": -1.0, "std": 1.2}

# Error patterns for validation testing
ERROR_PATTERNS = {
    "negative_samples": r"Number of samples must be positive",
    "negative_features": r"Number of features must be positive",
    "negative_std": r"Standard deviation .* must be positive",
    "missing_parameters": r"Feature .* must have 'mean' and 'std' parameters",
    "invalid_feature_type": r"Invalid feature type",
    "no_data": r"No data generated. Call generate_features\(\) first.",
    "feature_not_found": r"Feature .* not found in data",
}

# Dataset configurations for different test scenarios
DATASET_CONFIGS = {
    "minimal": {"samples": 10, "features": 2},
    "small": {"samples": 50, "features": 3},
    "medium": {"samples": 100, "features": 5},
    "large": {"samples": 1000, "features": 10},
}

# Reproducibility seeds for testing
REPRODUCIBILITY_SEEDS = {
    "default": 42,
    "alternative": 123,
    "third": 456,
}

# Validation ranges for testing
VALIDATION_RANGES = {
    "samples": {"min": 1, "max": 100000},
    "features": {"min": 1, "max": 1000},
    "std": {"min": 0.001, "max": 1000.0},
}
