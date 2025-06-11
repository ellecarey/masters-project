import torch
from torch.utils.data import Dataset
import pandas as pd


class TabularDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular data from a pandas DataFrame.
    """

    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
