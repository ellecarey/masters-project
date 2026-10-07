import matplotlib

# Use a non-interactive backend so tests do not open plot windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training_module.utils import (
    plot_final_metrics,
    plot_training_history,
)


def assert_pdf_created(path):
    assert path.is_file(), f"Missing PDF: {path}"
    assert path.stat().st_size > 0, f"Empty PDF: {path}"
    with path.open("rb") as file:
        assert file.read(5) == b"%PDF-", f"Not a PDF: {path}"


@pytest.mark.parametrize(
    ("trial_number", "expected_suffix"),
    [
        (None, ""),
        (0, "_trial0"),
        (7, "_trial7"),
    ],
)
def test_final_metrics_writes_expected_pdfs(tmp_path, trial_number, expected_suffix):
    # Identity returns these known logits unchanged.
    model = torch.nn.Identity()
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=2,
        shuffle=False,
    )

    # Plot tests should not require a system LaTeX installation.
    with plt.rc_context({"text.usetex": False}):
        try:
            plot_final_metrics(
                model=model,
                test_loader=loader,
                device="cpu",
                model_name="test_model",
                trial_number=trial_number,
                output_dir=tmp_path,
                subtitle="Temporary plotting test",
            )
        finally:
            plt.close("all")

    expected_names = {
        f"test_model_roc_curve{expected_suffix}.pdf",
        f"test_model_confusion_matrix{expected_suffix}.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names

    for filename in expected_names:
        assert_pdf_created(tmp_path / filename)


def test_training_history_writes_pdf(tmp_path):
    history = {
        "train_loss": [0.7, 0.5],
        "val_loss": [0.75, 0.55],
        "train_acc_epoch_end": [0.5, 0.75],
        "val_acc": [0.5, 0.7],
    }

    with plt.rc_context({"text.usetex": False}):
        try:
            plot_training_history(
                history=history,
                experiment_name="test_training_history",
                output_dir=tmp_path,
                subtitle="Temporary training-history test",
            )
        finally:
            plt.close("all")

    assert_pdf_created(tmp_path / "test_training_history.pdf")
