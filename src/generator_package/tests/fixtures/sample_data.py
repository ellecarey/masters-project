"""
Sample data and constants for testing the GaussianDataGenerator.
"""

import numpy as np

SAMPLE_FEATURE_PARAMS = {
    "test_feature_1": {"mean": 0, "std": 1},
    "test_feature_2": {"mean": 10, "std": 3},
    "test_feature_3": {"mean": -5, "std": 0.5},
    "test_feature_4": {"mean": 2.5, "std": 1.5},
}

EXTREME_FEATURE_PARAMS = {
    "extreme_large": {"mean": 1e6, "std": 1e3},
    "extreme_small": {"mean": -1e6, "std": 1e-3},
    "extreme_precision": {"mean": 0.123456789, "std": 0.987654321},
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
    "continuous_1": "continuous",
    "discrete_1": "discrete",
    "continuous_2": "continuous",
}

INVALID_FEATURE_TYPES = {
    "invalid_type_1": "invalid",
    "invalid_type_2": "categorical",
    "invalid_type_3": 123,
    "invalid_type_4": None,
}

# Weight constants for target variable creation
TEST_WEIGHTS = {
    "simple": [1.0, -0.5, 0.3],
    "complex": [2.5, -1.8, 0.7, -0.2, 1.1],
    "single": [1.0],
    "zeros": [0.0, 0.0, 0.0],
    "large": [100.0, -50.0, 25.0],
    "small": [0.001, -0.002, 0.003],
}

# Perturbation scale constants
PERTURBATION_SCALES = {"small": 0.01, "medium": 0.1, "large": 0.5, "very_large": 1.0}

# Function types for target creation
FUNCTION_TYPES = ["linear", "polynomial", "logistic"]

# Noise levels for target creation
NOISE_LEVELS = {"none": 0.0, "low": 0.05, "medium": 0.1, "high": 0.3, "very_high": 0.5}

# Sample sizes for different test scenarios
SAMPLE_SIZES = {
    "tiny": 5,
    "small": 50,
    "medium": 500,
    "large": 5000,
    "very_large": 50000,
}

# Feature counts for different test scenarios
FEATURE_COUNTS = {"single": 1, "few": 3, "medium": 10, "many": 50, "very_many": 100}

# Random states for reproducibility testing
RANDOM_STATES = [42, 123, 999, 2023, 12345]

# Expected value ranges for validation
EXPECTED_RANGES = {
    "default_mean_range": (-5, 5),
    "default_std_range": (0.5, 3),
    "probability_range": (0, 1),
    "binary_values": {0, 1},
}

# Test data for edge cases
EDGE_CASE_PARAMS = {
    "minimum_samples": 1,
    "minimum_features": 1,
    "maximum_reasonable_samples": 100000,
    "maximum_reasonable_features": 1000,
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "correlation_threshold": 0.95,  # For reproducibility tests
    "statistical_tolerance": 0.1,  # For mean/std validation
    "probability_tolerance": 0.05,  # For probability-based tests
}

# Error message patterns for testing
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

# Performance benchmarks (for performance testing)
PERFORMANCE_BENCHMARKS = {
    "small_dataset_time": 1.0,  # seconds
    "medium_dataset_time": 5.0,  # seconds
    "large_dataset_time": 30.0,  # seconds
}
