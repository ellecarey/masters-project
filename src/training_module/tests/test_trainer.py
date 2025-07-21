import torch
from torch import nn
from training_module.trainer import train_model


def test_train_model_updates_weights(initialised_model, sample_data_loader):
    """
    Tests that the model's weights change after one training epoch,
    indicating that backpropagation and optimiser steps are working.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(initialised_model.parameters(), lr=0.001)

    # Store initial weights of the first layer
    initial_weights = initialised_model.layer1.weight.clone().detach()

    # Train for one epoch
    train_model(
        model=initialised_model,
        train_loader=sample_data_loader,
        validation_loader=sample_data_loader,  # Can reuse for this test
        criterion=criterion,
        optimiser=optimiser,
        epochs=1,
    )

    # Get weights after training
    updated_weights = initialised_model.layer1.weight.clone().detach()

    # Check that the weights are not the same as the initial ones
    assert not torch.equal(initial_weights, updated_weights)


def test_train_model_returns_trained_model(initialised_model, sample_data_loader):
    """Tests if the function returns a model instance."""
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(initialised_model.parameters(), lr=0.001)

    trained_model = train_model(
        model=initialised_model,
        train_loader=sample_data_loader,
        validation_loader=sample_data_loader,
        criterion=criterion,
        optimiser=optimiser,
        epochs=1,
    )

    assert isinstance(trained_model, torch.nn.Module)
