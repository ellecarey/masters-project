import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import optuna

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: str,
    trial: optuna.trial.Trial = None, # Add trial as an optional argument
):
    """
    Model training and validation loops. Now supports Optuna pruning.
    """
    model.to(device)
    print(f"Starting model training on device: '{device}'...")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        )

        for features, labels in train_progress_bar:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
            train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)

        # --- Validation loop ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = correct_val / total_val
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # --- Pruning Logic ---
        if trial:
            # Report the validation accuracy to Optuna
            trial.report(val_accuracy, epoch)

            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

    print("Training finished.")
    return model, history