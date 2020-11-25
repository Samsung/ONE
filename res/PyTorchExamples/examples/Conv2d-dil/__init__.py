import torch
import torch.nn as nn


# model
class net_Conv2dDil(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Conv2d(2, 2, 1, dilation=2)

    def forward(self, input):
        return self.op(input)


_model_ = net_Conv2dDil()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
