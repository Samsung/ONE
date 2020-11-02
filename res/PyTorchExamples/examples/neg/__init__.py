import torch
import torch.nn as nn


# model
class net_neg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.neg(input)


_model_ = net_neg()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
