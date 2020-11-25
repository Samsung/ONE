import torch
import torch.nn as nn


# model
class net_min(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.min(input, 0, True)


_model_ = net_min()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
