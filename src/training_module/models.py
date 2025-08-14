import torch
import torch.nn as nn
from typing import Dict, Any

class mlp_001(nn.Module):
    """
    A simple Multi-Layer Perceptron with one hidden layer.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(mlp_001, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out
        
# class mlp_002(nn.Module):
#     """
#     A simple Multi-Layer Perceptron with one hidden layer.
#     """
#     def __init__(self, input_size: int, hidden_size: int, output_size: int):
#         super(mlp_002, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.relu(out)
#         out = self.layer2(out)
#         return out
        
# --- The Model Factory ---
def get_model(model_name: str, model_params: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate a model based on its name.
    """
    if model_name == "mlp_001":
        return mlp_001(**model_params)
    # elif model_name == "mlp_002":
    #     return mlp_002(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

