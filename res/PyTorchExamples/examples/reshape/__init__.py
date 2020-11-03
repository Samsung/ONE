import torch
import torch.nn as nn


# model
class net_reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.reshape(input, (2, 9))


_model_ = net_reshape()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
