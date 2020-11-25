import torch
import torch.nn as nn


# model
class net_pow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.pow(input, 5)


_model_ = net_pow()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
