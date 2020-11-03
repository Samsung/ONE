import torch
import torch.nn as nn


# model
class net_split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.split(input, 2)


_model_ = net_split()

# dummy input for onnx generation
_dummy_ = torch.randn(2, 2, 3, 3)
