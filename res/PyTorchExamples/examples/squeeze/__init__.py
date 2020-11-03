import torch
import torch.nn as nn


# model
class net_squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.squeeze(input)


_model_ = net_squeeze()

# dummy input for onnx generation
_dummy_ = torch.randn(2, 1, 3, 3)
