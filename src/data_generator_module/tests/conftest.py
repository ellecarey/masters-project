"""
Updated pytest configuration and fixtures for feature-based classification.
"""

import pytest
import pandas as pd
from data_generator_module.gaussian_data_generator import GaussianDataGenerator


@pytest.fixture
def sample_feature_params():
    """Sample feature parameters for testing"""
    return {
        "feature_0": {"mean": 0.0, "std": 1.0},
        "feature_1": {"mean": 10.0, "std": 2.0},
        "feature_2": {"mean": -5.0, "std": 1.5},
    }


@pytest.fixture
def sample_feature_types():
    """Sample feature types for testing"""
    return {
        "feature_0": "continuous",
        "feature_1": "discrete",
        "feature_2": "continuous",
    }


@pytest.fixture
def basic_generator():
    """Basic generator instance for testing"""
    return GaussianDataGenerator(n_samples=100, n_features=3, random_state=42)


@pytest.fixture
def generator_with_sample_data(sample_feature_params, sample_feature_types):
    """Generator with sample data already generated"""
    gen = GaussianDataGenerator(
        n_samples=100, n_features=len(sample_feature_params), random_state=42
    )
    gen.generate_features(sample_feature_params, sample_feature_types)
    return gen


@pytest.fixture
def generator_factory():
    """Factory to create generators with custom parameters"""

    def _create_generator(n_samples=100, n_features=5, random_state=42):
        return GaussianDataGenerator(
            n_samples=n_samples, n_features=n_features, random_state=random_state
        )

    return _create_generator


@pytest.fixture
def reproducible_generators():
    """Two generators with same seed for reproducibility testing"""
    gen1 = GaussianDataGenerator(n_samples=100, n_features=3, random_state=42)
    gen2 = GaussianDataGenerator(n_samples=100, n_features=3, random_state=42)
    return gen1, gen2


@pytest.fixture
def different_seed_generators():
    """Two generators with different seeds"""
    gen1 = GaussianDataGenerator(n_samples=100, n_features=3, random_state=42)
    gen2 = GaussianDataGenerator(n_samples=100, n_features=3, random_state=123)
    return gen1, gen2
