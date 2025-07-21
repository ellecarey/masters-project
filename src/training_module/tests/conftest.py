import pytest
import pandas as pd
from torch.utils.data import DataLoader

from training_module.mlp_model import MLP
from training_module.dataset import TabularDataset


@pytest.fixture
def sample_training_data():
    """Creates a sample DataFrame for training tests."""
    data = {
        "feature_0": [i for i in range(100)],
        "feature_1": [i * 2 for i in range(100)],
        "feature_2": [i * -1 for i in range(100)],
        "target": [i % 2 for i in range(100)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_tabular_dataset(sample_training_data):
    """Creates a TabularDataset instance from sample data."""
    features = sample_training_data.drop(columns=["target"])
    labels = sample_training_data["target"]
    return TabularDataset(features, labels)


@pytest.fixture
def sample_data_loader(sample_tabular_dataset):
    """Creates a DataLoader instance for testing."""
    return DataLoader(sample_tabular_dataset, batch_size=10, shuffle=True)


@pytest.fixture
def initialised_model():
    """Initializes a standard MLP model for testing."""
    # input_size=3 corresponds to the three features in sample_training_data
    return MLP(input_size=3, hidden_size=16, output_size=1)
