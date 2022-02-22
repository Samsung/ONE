import torch
import torch.nn as nn


# model
class net_clamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.clamp(input, 0, 10)


_model_ = net_clamp()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
