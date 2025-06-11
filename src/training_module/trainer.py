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
):
    """model training and validation loops."""
    print("Starting model training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        )
        for features, labels in train_progress_bar:
            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in validation_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    print("Training finished.")
    return model
