"""Conservative, per-process ownership records for disposable CSV datasets.

No persistent ownership is inferred from filenames. Interrupted runs leave their
files in place. Do not run concurrent writers against the same output directory.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import stat

import yaml


@dataclass(frozen=True)
class FileStamp:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def file_stamp(path: Path) -> FileStamp:
    """Reject symlinks and non-regular files; detect ordinary file changes."""
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    return FileStamp(
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    )


@dataclass(frozen=True)
class GeneratedDataset:
    config_path: Path
    csv_path: Path
    stamp: FileStamp


@dataclass(frozen=True)
class EvaluatedDataset:
    csv_path: Path
    csv_stamp: FileStamp
    metrics_path: Path
    metrics_stamp: FileStamp


def dataset_path(project_root: Path, config: dict, filename_base: str) -> Path:
    directory = Path(config.get("output_settings", {}).get("data_dir", "data/"))
    if not directory.is_absolute():
        directory = Path(project_root) / directory
    # Resolve the parent only: do not follow a symlink at the CSV itself.
    return directory.resolve() / f"{filename_base}_dataset.csv"


def write_config_without_overwrite(path: Path, config: dict) -> None:
    """Reuse identical YAML; refuse to replace a different existing config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    except FileExistsError:
        file_stamp(path)
        with path.open(encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if existing != config:
            raise FileExistsError(f"Refusing to overwrite a different config: {path}")


def save_new_dataset(data, csv_path: Path, config_path: Path) -> GeneratedDataset:
    """Write exclusively; a pre-existing CSV can never become run-owned."""
    if data is None:
        raise ValueError("No generated data to save")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also refuses existing symlinks. On write failure leave
    # the partial file for inspection; no ownership record is issued.
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        data.to_csv(handle, index=False)
    return GeneratedDataset(
        Path(config_path).absolute(), csv_path.absolute(), file_stamp(csv_path)
    )


def validate_metrics(path: Path) -> None:
    """Check required finite evaluation metrics, not historical provenance."""
    file_stamp(path)
    with Path(path).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    for name in ("Test Loss (BCE)", "Accuracy", "F1-Score", "Precision", "Recall", "AUC"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Missing or non-numeric metric {name!r} in {path}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metric {name!r} in {path}")
        if name != "Test Loss (BCE)" and value > 1:
            raise ValueError(f"Out-of-range metric {name!r} in {path}")


def evaluation_config_paths(generated):
    """Training datasets are retained and never candidates for evaluation cleanup."""
    return [
        item.config_path for item in generated
        if not item.csv_path.name.endswith("_training_dataset.csv")
    ]


def cleanup_evaluated_datasets(generated, evaluated):
    """Delete the intersection of this run's creations and fresh evaluations.

Validate every candidate before any deletion. Skipped/cached evaluations do not
authorize deletion. Stat stamps detect changes, but are not a cross-process lock.
    """
    candidates = []
    for item in generated:
        if item.csv_path.name.endswith("_training_dataset.csv"):
            continue
        receipt = evaluated.get(item.csv_path)
        if receipt is None:
            print(f"Keeping dataset without fresh evaluation: {item.csv_path}")
            continue
        if receipt.csv_path != item.csv_path or receipt.csv_stamp != item.stamp:
            raise RuntimeError(f"Dataset ownership/evaluation mismatch: {item.csv_path}")
        if file_stamp(item.csv_path) != item.stamp:
            raise RuntimeError(f"Dataset changed; refusing cleanup: {item.csv_path}")
        if file_stamp(receipt.metrics_path) != receipt.metrics_stamp:
            raise RuntimeError(f"Metrics changed; refusing cleanup: {receipt.metrics_path}")
        validate_metrics(receipt.metrics_path)
        candidates.append(item)

    deleted = []
    for item in candidates:
        # Recheck immediately before deletion; do not silently swallow I/O errors.
        if file_stamp(item.csv_path) != item.stamp:
            raise RuntimeError(f"Dataset changed during cleanup: {item.csv_path}")
        item.csv_path.unlink()
        deleted.append(item.csv_path)
        print(f"Removed newly generated, evaluated dataset: {item.csv_path}")
    return deleted
