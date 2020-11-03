import torch
import torch.nn as nn


# model
class net_permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.permute(3, 0, 2, 1)


_model_ = net_permute()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
