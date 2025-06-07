"""
Test package for generator_package.

This package contains comprehensive unit tests for:
- GaussianDataGenerator class
- DataGeneratorValidators
- Utility functions
- Integration tests and edge cases

The tests are organised into separate modules:
- test_gaussian_data_generator.py: Main class functionality tests
- test_validators.py: Parameter validation tests
- test_utils.py: Utility function tests

Shared fixtures and test data are available in:
- conftest.py: Pytest fixtures and configuration
- fixtures/: Test data constants and sample data
"""

from .fixtures.sample_data import (
    SAMPLE_FEATURE_PARAMS,
    INVALID_FEATURE_PARAMS,
    ERROR_PATTERNS,
    TARGET_WEIGHTS,
    FUNCTION_TYPES,
)

__all__ = [
    "SAMPLE_FEATURE_PARAMS",
    "INVALID_FEATURE_PARAMS",
    "ERROR_PATTERNS",
    "TARGET_WEIGHTS",
    "FUNCTION_TYPES",
]
