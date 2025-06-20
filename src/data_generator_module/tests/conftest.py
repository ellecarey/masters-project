"""
pytest configuration and shared fixtures
"""

import pytest
from data_generator_module.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import (
    SAMPLE_FEATURE_PARAMS,
    SAMPLE_FEATURE_TYPES,
    REPRODUCIBILITY_SEEDS,
    DATASET_CONFIGS,
)


@pytest.fixture
def sample_config():
    """
    A shared, standard configuration fixture for data generator tests.
    This fixture is now available to all tests in this directory.
    """
    return {
        "dataset_settings": {"n_samples": 1000, "n_initial_features": 5},
        "add_features": {"n_new_features": 2},
        "perturbation": {"perturbation_type": "gaussian", "scale": 0.15},
        "create_target": {"function_type": "polynomial"},
    }


@pytest.fixture
def basic_generator():
    """Basic generator instance for testing"""
    return GaussianDataGenerator(
        n_samples=100, n_features=3, random_state=REPRODUCIBILITY_SEEDS[0]
    )


@pytest.fixture
def generator_with_sample_data():
    """Generator with data using sample parameters from fixtures"""
    gen = GaussianDataGenerator(
        n_samples=100,
        n_features=len(SAMPLE_FEATURE_PARAMS),
        random_state=REPRODUCIBILITY_SEEDS[0],
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


@pytest.fixture
def standard_test_config():
    """Standard test configuration from DATASET_CONFIGS"""
    return DATASET_CONFIGS["unit_test_standard"]


@pytest.fixture
def test_seeds():
    """Reproducibility seeds for testing"""
    return REPRODUCIBILITY_SEEDS


@pytest.fixture
def generator_factory():
    """Factory fixture to create generators with custom parameters"""

    def _create_generator(n_samples=100, n_features=5, random_state=None):
        if random_state is None:
            random_state = REPRODUCIBILITY_SEEDS[0]
        return GaussianDataGenerator(
            n_samples=n_samples, n_features=n_features, random_state=random_state
        )

    return _create_generator


@pytest.fixture
def reproducible_generators(standard_test_config):
    """Fixture providing two generators with same seed for reproducibility testing"""
    seed = REPRODUCIBILITY_SEEDS[0]

    gen1 = GaussianDataGenerator(
        n_samples=standard_test_config["samples"],
        n_features=len(SAMPLE_FEATURE_PARAMS),
        random_state=seed,
    )

    gen2 = GaussianDataGenerator(
        n_samples=standard_test_config["samples"],
        n_features=len(SAMPLE_FEATURE_PARAMS),
        random_state=seed,
    )

    return gen1, gen2


@pytest.fixture
def different_seed_generators(standard_test_config):
    """Fixture providing two generators with different seeds"""
    gen1 = GaussianDataGenerator(
        n_samples=standard_test_config["samples"],
        n_features=len(SAMPLE_FEATURE_PARAMS),
        random_state=REPRODUCIBILITY_SEEDS[0],
    )

    gen2 = GaussianDataGenerator(
        n_samples=standard_test_config["samples"],
        n_features=len(SAMPLE_FEATURE_PARAMS),
        random_state=REPRODUCIBILITY_SEEDS[1],
    )

    return gen1, gen2


# Performance testing fixture
@pytest.fixture
def large_dataset():
    """Larger dataset for performance testing using config values"""
    config = DATASET_CONFIGS.get(
        "performance_benchmark", {"samples": 1000, "features": 10}
    )
    gen = GaussianDataGenerator(
        n_samples=config["samples"],
        n_features=config["features"],
        random_state=REPRODUCIBILITY_SEEDS[0],
    )

    # Create default parameters for the larger dataset
    default_params = {}
    default_types = {}
    for i in range(config["features"]):
        feature_name = f"feature_{i}"
        default_params[feature_name] = {"mean": 0.0, "std": 1.0}
        default_types[feature_name] = "continuous"

    gen.generate_features(default_params, default_types)
    return gen
