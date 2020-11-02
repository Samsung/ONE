import torch
import torch.nn as nn


# model
class net_ne(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.ne(inputs[0], inputs[1])


_model_ = net_ne()

# dummy input for onnx generation
_dummy_ = [torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
