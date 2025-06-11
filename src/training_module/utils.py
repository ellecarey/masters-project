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
