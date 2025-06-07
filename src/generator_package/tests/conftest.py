"""
pytest configuration and shared fixtures
"""

import pytest
from generator_package.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES


@pytest.fixture
def basic_generator():
    """basic generator instance for testing"""
    return GaussianDataGenerator(n_samples=100, n_features=3, random_state=42)


@pytest.fixture
def generator_with_data():
    """Generator instance with pre-generated data"""
    gen = GaussianDataGenerator(n_samples=50, n_features=2, random_state=123)
    gen.generate_features()
    return gen


@pytest.fixture
def generator_with_sample_data():
    """generator with data using sample parameters from fixtures"""
    gen = GaussianDataGenerator(
        n_samples=100, n_features=len(SAMPLE_FEATURE_PARAMS), random_state=42
    )
    gen.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)
    return gen


@pytest.fixture
def sample_feature_params():
    """Sample feature parameters from fixtures."""
    return SAMPLE_FEATURE_PARAMS.copy()


@pytest.fixture
def sample_feature_types():
    """Sample feature types from fixtures."""
    return SAMPLE_FEATURE_TYPES.copy()


# Performance testing fixture
@pytest.fixture
@pytest.mark.slow
def large_dataset():
    """Larger dataset for performance testing."""
    from .fixtures.sample_data import DATASET_CONFIGS

    config = DATASET_CONFIGS["performance_benchmark"]
    gen = GaussianDataGenerator(
        n_samples=config["samples"],
        n_features=config["features"],
        random_state=42,
    )
    gen.generate_features()
    return gen
