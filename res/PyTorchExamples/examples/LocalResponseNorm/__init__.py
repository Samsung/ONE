import torch
import torch.nn as nn


# model
class net_LocalResponseNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.LocalResponseNorm(1)

    def forward(self, input):
        return self.op(input)


_model_ = net_LocalResponseNorm()
# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 4, 4)
