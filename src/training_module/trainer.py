import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import optuna
import copy
from typing import Optional
from sklearn.metrics import roc_auc_score, accuracy_score 
import numpy as np

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: str,
    validation_loader: DataLoader = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    trial: Optional[optuna.trial.Trial] = None,
    early_stopping_enabled: bool = False,
    patience: int = 10,
    min_delta: float = 0.0001,
    verbose: bool = True
): 
    """
    Trains a PyTorch model with support for validation, early stopping, and Optuna pruning.

    Returns:
        A tuple containing the trained model, a history dictionary of metrics,
        and the best performing epoch number.
    """
    model.to(device)
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc_batch": [], "train_acc_epoch_end": [], "val_acc": [],
        "val_auc": []
    }

    # --- Initialize tracking variables outside of any conditional blocks. ---
    # This ensures we always track the best epoch, even if early stopping isn't triggered.
    best_val_loss = float('inf')
    best_epoch = 1  # Default to epoch 1
    best_model_state = copy.deepcopy(model.state_dict()) # Save initial state
    epochs_no_improve = 0

    epoch_iterator = tqdm(range(epochs), desc="Training", leave=False, disable=not verbose)

    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            
            # For accuracy calculation
            train_preds.extend(torch.round(torch.sigmoid(outputs)).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_acc_epoch_end"].append(accuracy_score(train_labels, train_preds))

        # --- Validation Step ---
        avg_val_loss, val_acc, val_auc = None, None, None
        if validation_loader:
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for features, labels in validation_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    val_loss += criterion(outputs, labels).item()
                    
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(validation_loader)
            val_acc = accuracy_score(np.array(val_labels).round(), np.array(val_preds).round())
            val_auc = roc_auc_score(val_labels, val_preds)

            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)

        # Update progress bar
        if verbose:
            postfix_data = {'Train Loss': f'{avg_train_loss:.4f}'}
            if avg_val_loss is not None:
                postfix_data['Val Loss'] = f'{avg_val_loss:.4f}'
                postfix_data['Val AUC'] = f'{val_auc:.4f}'
            epoch_iterator.set_postfix(postfix_data)
        
        if scheduler and avg_val_loss is not None:
            scheduler.step(avg_val_loss)

        # --- Always track best epoch, but only use patience if enabled. ---
        if avg_val_loss is not None:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                # Only update model state and patience counter if early stopping is on
                if early_stopping_enabled:
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy(model.state_dict())
            elif early_stopping_enabled:
                epochs_no_improve += 1

        # Check for early stopping (patience-based)
        if early_stopping_enabled and epochs_no_improve >= patience:
            if verbose:
                epoch_iterator.write(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

        # Check for Optuna pruning
        if trial:
            if avg_val_loss is not None:
                trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                # --- COMBINED FIX 3: Save the best epoch found so far before pruning. ---
                trial.set_user_attr("best_epoch", best_epoch)
                raise optuna.TrialPruned()

    # --- After the loop ---
    if early_stopping_enabled and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Always return the best epoch that was tracked throughout the run
    return model, history, best_epoch