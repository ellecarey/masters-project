"""
Updated pytest configuration and fixtures for feature-based classification with separate distributions.
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
def signal_noise_config():
    """Configuration for separate signal/noise distributions"""
    return {
        "signal_features": {
            "feature_0": {"mean": 15.0, "std": 1.0},
            "feature_1": {"mean": -2.0, "std": 0.5},
            "feature_2": {"mean": 8.0, "std": 1.2},
        },
        "noise_features": {
            "feature_0": {"mean": 5.0, "std": 2.0},
            "feature_1": {"mean": 8.0, "std": 1.8},
            "feature_2": {"mean": -5.0, "std": 0.8},
        },
        "feature_types": {
            "feature_0": "discrete",
            "feature_1": "discrete",
            "feature_2": "continuous",
        },
    }


@pytest.fixture
def high_separation_config():
    """Configuration with high separation between signal and noise"""
    return {
        "signal_features": {
            "feature_0": {"mean": 20.0, "std": 1.0},
            "feature_1": {"mean": -15.0, "std": 1.0},
            "feature_2": {"mean": 10.0, "std": 1.0},
        },
        "noise_features": {
            "feature_0": {"mean": 0.0, "std": 1.0},
            "feature_1": {"mean": 10.0, "std": 1.0},
            "feature_2": {"mean": -10.0, "std": 1.0},
        },
        "feature_types": {
            "feature_0": "discrete",
            "feature_1": "discrete",
            "feature_2": "continuous",
        },
    }


@pytest.fixture
def low_separation_config():
    """Configuration with low separation between signal and noise"""
    return {
        "signal_features": {
            "feature_0": {"mean": 5.0, "std": 3.0},
            "feature_1": {"mean": -2.0, "std": 3.0},
            "feature_2": {"mean": 3.0, "std": 3.0},
        },
        "noise_features": {
            "feature_0": {"mean": 3.0, "std": 3.0},
            "feature_1": {"mean": 0.0, "std": 3.0},
            "feature_2": {"mean": 1.0, "std": 3.0},
        },
        "feature_types": {
            "feature_0": "discrete",
            "feature_1": "discrete",
            "feature_2": "continuous",
        },
    }


@pytest.fixture
def basic_generator():
    """Basic generator instance for testing"""
    return GaussianDataGenerator(n_samples=100, n_features=3, random_state=42)


@pytest.fixture
def large_generator():
    """Larger generator instance for statistical testing"""
    return GaussianDataGenerator(n_samples=1000, n_features=3, random_state=42)


@pytest.fixture
def generator_with_sample_data(sample_feature_params, sample_feature_types):
    """Generator with sample data already generated - LEGACY for backward compatibility"""
    gen = GaussianDataGenerator(
        n_samples=100, n_features=len(sample_feature_params), random_state=42
    )
    gen.generate_features(sample_feature_params, sample_feature_types)
    return gen


@pytest.fixture
def generator_with_signal_noise_data(basic_generator, signal_noise_config):
    """Generator with signal/noise data already generated"""
    basic_generator.create_feature_based_signal_noise_classification(
        **signal_noise_config
    )
    return basic_generator


@pytest.fixture
def generator_factory():
    """Factory to create generators with custom parameters"""

    def _create_generator(n_samples=100, n_features=3, random_state=42):
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


@pytest.fixture
def edge_case_configs():
    """Dictionary of edge case configurations for testing"""
    return {
        "identical_distributions": {
            "signal_features": {"feature_0": {"mean": 0.0, "std": 1.0}},
            "noise_features": {"feature_0": {"mean": 0.0, "std": 1.0}},
            "feature_types": {"feature_0": "continuous"},
        },
        "extreme_separation": {
            "signal_features": {"feature_0": {"mean": 1000.0, "std": 1.0}},
            "noise_features": {"feature_0": {"mean": -1000.0, "std": 1.0}},
            "feature_types": {"feature_0": "continuous"},
        },
        "very_small_std": {
            "signal_features": {"feature_0": {"mean": 10.0, "std": 0.001}},
            "noise_features": {"feature_0": {"mean": -10.0, "std": 0.001}},
            "feature_types": {"feature_0": "continuous"},
        },
    }


@pytest.fixture
def mixed_feature_types_config():
    """Configuration with mixed continuous and discrete features"""
    return {
        "signal_features": {
            "feature_0": {"mean": 10.0, "std": 2.0},  # continuous
            "feature_1": {"mean": 5.0, "std": 1.0},  # discrete
            "feature_2": {"mean": -3.0, "std": 1.5},  # continuous
            "feature_3": {"mean": 8.0, "std": 1.2},  # discrete
            "feature_4": {"mean": 0.0, "std": 2.0},  # continuous
        },
        "noise_features": {
            "feature_0": {"mean": 2.0, "std": 2.0},  # continuous
            "feature_1": {"mean": -2.0, "std": 1.0},  # discrete
            "feature_2": {"mean": 7.0, "std": 1.5},  # continuous
            "feature_3": {"mean": -5.0, "std": 1.2},  # discrete
            "feature_4": {"mean": 10.0, "std": 2.0},  # continuous
        },
        "feature_types": {
            "feature_0": "continuous",
            "feature_1": "discrete",
            "feature_2": "continuous",
            "feature_3": "discrete",
            "feature_4": "continuous",
        },
    }


@pytest.fixture
def performance_test_config():
    """Configuration for performance testing with many features"""
    signal_features = {}
    noise_features = {}
    feature_types = {}

    for i in range(20):  # 20 features for performance testing
        signal_features[f"feature_{i}"] = {"mean": i * 2.0, "std": 1.0}
        noise_features[f"feature_{i}"] = {"mean": i * -1.0, "std": 1.0}
        feature_types[f"feature_{i}"] = "continuous" if i % 2 == 0 else "discrete"

    return {
        "signal_features": signal_features,
        "noise_features": noise_features,
        "feature_types": feature_types,
    }
