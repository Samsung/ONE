import torch
import torch.nn as nn


# model
class net_normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.normalize(input, p=2.0, dim=3, eps=1e-12)


_model_ = net_normalize()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
