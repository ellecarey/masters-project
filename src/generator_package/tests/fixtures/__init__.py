"""
Test fixtures package for generator_package tests.

This package provides:
- Sample data constants for testing
- Invalid data for error testing
- Test parameters and configurations
- Shared test utilities

Available modules:
- sample_data: Test data constants and parameters
"""

from .sample_data import (
    SAMPLE_FEATURE_PARAMS,
    EXTREME_FEATURE_PARAMS,
    INVALID_FEATURE_PARAMS,
    SAMPLE_FEATURE_TYPES,
    INVALID_FEATURE_TYPES,
    TARGET_WEIGHTS,
    PERTURBATION_LEVELS,
    FUNCTION_TYPES,
    TARGET_NOISE_LEVELS,
    ERROR_PATTERNS,
    REPRODUCIBILITY_SEEDS,
    VALIDATION_RANGES,
)


__all__ = [
    "SAMPLE_FEATURE_PARAMS",
    "EXTREME_FEATURE_PARAMS",
    "INVALID_FEATURE_PARAMS",
    "SAMPLE_FEATURE_TYPES",
    "INVALID_FEATURE_TYPES",
    "TARGET_WEIGHTS",
    "PERTURBATION_LEVELS",
    "FUNCTION_TYPES",
    "TARGET_NOISE_LEVELS",
    "ERROR_PATTERNS",
    "REPRODUCIBILITY_SEEDS",
    "VALIDATION_RANGES",
]
