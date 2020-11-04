import torch
import torch.nn as nn


# model
class net_zeros_like(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.zeros_like(input)


_model_ = net_zeros_like()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
