import torch
import torch.nn as nn


# model
class net_add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.add(inputs[0], inputs[1])


_model_ = net_add()

# dummy input for onnx generation
_dummy_ = [torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
