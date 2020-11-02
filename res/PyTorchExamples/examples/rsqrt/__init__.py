import torch
import torch.nn as nn


# model
class net_rsqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.rsqrt(input)


_model_ = net_rsqrt()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
