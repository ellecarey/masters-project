import torch
from training_module.mlp_model import MLP


def test_mlp_initialisation(initialised_model):
    """Tests if the MLP model layers are created correctly."""
    assert isinstance(initialised_model.layer1, torch.nn.Linear)
    assert initialised_model.layer1.in_features == 3
    assert initialised_model.layer1.out_features == 16

    assert isinstance(initialised_model.layer2, torch.nn.Linear)
    assert initialised_model.layer2.in_features == 16
    assert initialised_model.layer2.out_features == 1


def test_mlp_forward_pass(initialised_model):
    """Tests the forward pass of the model."""
    # Create a dummy input tensor with the correct shape (batch_size, num_features)
    input_tensor = torch.randn(10, 3)  # Batch of 10 samples
    output = initialised_model(input_tensor)

    # Output shape should be (batch_size, output_size)
    assert output.shape == (10, 1)
