import torch
import torch.nn as nn


# model
class net_AdaptiveMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.AdaptiveMaxPool2d(1)

    def forward(self, input):
        return self.op(input)


_model_ = net_AdaptiveMaxPool2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
