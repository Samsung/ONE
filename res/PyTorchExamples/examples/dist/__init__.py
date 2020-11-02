import torch
import torch.nn as nn


# model
class net_linarg_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.dist(inputs[0], inputs[1])


_model_ = net_linarg_norm()
# dummy input for onnx generation
_dummy_ = [torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
