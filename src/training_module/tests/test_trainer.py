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
        device=torch.device("cpu"),
    )

    # Get weights after training
    updated_weights = initialised_model.layer1.weight.clone().detach()

    # Check that the weights are not the same as the initial ones
    assert not torch.equal(initial_weights, updated_weights)


def test_train_model_returns_model_history_and_best_epoch(
    initialised_model, sample_data_loader
):
    """Tests the documented return values after one training epoch."""
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(initialised_model.parameters(), lr=0.001)

    trained_model, history, best_epoch = train_model(
        model=initialised_model,
        train_loader=sample_data_loader,
        validation_loader=sample_data_loader,
        criterion=criterion,
        optimiser=optimiser,
        epochs=1,
        device=torch.device("cpu"),
    )

    assert isinstance(trained_model, torch.nn.Module)
    assert isinstance(history, dict)

    # One training epoch with validation should record one of each metric.
    for metric in (
        "train_loss",
        "train_acc_epoch_end",
        "val_loss",
        "val_acc",
        "val_auc",
    ):
        assert metric in history
        assert len(history[metric]) == 1

    # With only one epoch, the best epoch must be epoch 1.
    assert best_epoch == 1
