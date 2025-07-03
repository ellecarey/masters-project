import random
import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For GPU operations, if you are using one
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global random seed set to {seed}")


def resolve_dataset_path(training_config, data_dir="data"):
    """
    Resolve the dataset path from the training configuration.

    """
    data_source = training_config.get("data_source", {})
    dataset_config = data_source.get("dataset_config")

    if not dataset_config:
        raise ValueError("No dataset_config specified in training configuration")

    # Construct path to the generated dataset
    dataset_filename = f"{dataset_config}_dataset.csv"
    dataset_path = os.path.join(data_dir, dataset_filename)

    # Verify the file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Make sure to run data generation with config '{dataset_config}' first."
        )

    return dataset_path


def load_yaml_config(config_path):
    """
    Load YAML configuration file for training.
    """
    import yaml

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Training configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing training configuration YAML: {e}")
