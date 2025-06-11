import torch.nn as nn


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron with one hidden layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out
