import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import optuna
import copy
from typing import Optional

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: Optional[DataLoader],
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    trial: optuna.trial.Trial = None,
    verbose: bool = False,
    early_stopping_enabled: bool = True,
    patience: int = 20,
    min_delta: float = 1e-5
    
):
    """
    Model training function that can run with or without a validation loader.
    If no validation_loader is provided, early stopping and validation-based
    schedulers are automatically disabled.
    """
    model.to(device)
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc_avg": [], "val_acc": [], "train_acc_epoch_end": []
    }

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    epoch_iterator = tqdm(range(epochs), desc="Training Progress", unit="epoch", disable=not verbose)

    for epoch in epoch_iterator:
        # --- Training Loop ---
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_train_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
        
        avg_epoch_train_loss = running_train_loss / len(train_loader)
        avg_epoch_train_acc = correct_train / total_train
        history["train_loss"].append(avg_epoch_train_loss)
        history["train_acc_avg"].append(avg_epoch_train_acc)

        # --- Post-Epoch Evaluation ---
        model.eval()
        avg_val_loss = None  # Initialize to None

        # MODIFIED: Evaluate on Validation Set only if it exists
        if validation_loader:
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

        # Evaluate on Training Set (End of Epoch Snapshot)
        post_train_loss, correct_post_train, total_post_train = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.round(torch.sigmoid(outputs))
                total_post_train += labels.size(0)
                correct_post_train += (preds == labels).sum().item()
        post_epoch_train_acc = correct_post_train / total_post_train
        history["train_acc_epoch_end"].append(post_epoch_train_acc)

        if verbose:
            postfix_data = {"Train Acc (End)": f"{post_epoch_train_acc:.4f}"}
            if validation_loader:
                postfix_data["Val Acc"] = f"{val_accuracy:.4f}"
            epoch_iterator.set_postfix(postfix_data)
        
        if scheduler and avg_val_loss is not None:
            scheduler.step(avg_val_loss)
            
        if early_stopping_enabled and avg_val_loss is not None:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                if verbose:
                    epoch_iterator.write(f"\nEarly stopping triggered after {patience} epochs.")
                break
        
        if trial:
            report_metric = val_accuracy if validation_loader else post_epoch_train_acc
            trial.report(report_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if early_stopping_enabled and best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_epoch_to_report = best_epoch if best_epoch > 0 else epochs
        
    return model, history, best_epoch