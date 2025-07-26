import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import optuna
import copy

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: str,
    trial: optuna.trial.Trial = None,
    patience: int = 5,
):
    """
    Model training and validation loops. This version is silent to ensure
    clean output when run by parallel workers.
    """
    model.to(device)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        # Use a non-visual progress bar when running as a worker
        for features, labels in train_loader:
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
        
        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            if best_model_state:
                model.load_state_dict(best_model_state)
            break

        # --- Pruning Logic ---
        if trial:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history