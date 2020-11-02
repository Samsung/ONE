import torch
import torch.nn as nn


# model
class net_slice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input[:, :, :, 0:1]


_model_ = net_slice()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
