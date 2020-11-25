import torch
import torch.nn as nn


# model
class net_logical_xor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.logical_xor(inputs[0], inputs[1])


_model_ = net_logical_xor()

# dummy input for onnx generation
_dummy_ = [torch.randn(1, 2, 3, 3).bool(), torch.randn(1, 2, 3, 3).bool()]

# Note: this model has problem when exporting to ONNX
