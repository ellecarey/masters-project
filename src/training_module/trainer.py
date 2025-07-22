import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: str,
):
    """Model training and validation loops, now with device awareness."""

    # --- Move the model to the specified device ---
    model.to(device)
    print(f"Starting model training on device: '{device}'...")

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        )

        for features, labels in train_progress_bar:
            # --- Move data batch to the same device as the model ---
            features, labels = features.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in validation_loader:
                # --- Move validation data to the device ---
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    print("Training finished.")
    return model
