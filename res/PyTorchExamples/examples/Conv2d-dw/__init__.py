import torch
import torch.nn as nn


# model
class net_Conv2dDW(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Conv2d(2, 2, 1, groups=2)

    def forward(self, input):
        return self.op(input)


_model_ = net_Conv2dDW()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
