import torch


def test_dataset_initialisation(sample_tabular_dataset, sample_training_data):
    """Tests if the dataset is initialised correctly with tensors."""
    assert isinstance(sample_tabular_dataset.features, torch.Tensor)
    assert isinstance(sample_tabular_dataset.labels, torch.Tensor)
    assert sample_tabular_dataset.features.dtype == torch.float32
    assert sample_tabular_dataset.labels.dtype == torch.float32


def test_dataset_length(sample_tabular_dataset, sample_training_data):
    """Tests the __len__ method."""
    assert len(sample_tabular_dataset) == len(sample_training_data)


def test_dataset_getitem(sample_tabular_dataset):
    """Tests if getting an item returns tensors of the correct shape."""
    features, label = sample_tabular_dataset[0]

    # Features should be a 1D tensor with length equal to the number of features
    assert features.shape == (3,)
    # Label should be a 1D tensor with a single element
    assert label.shape == (1,)
