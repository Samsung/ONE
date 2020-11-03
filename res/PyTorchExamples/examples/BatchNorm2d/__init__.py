import torch
import torch.nn as nn


# model
class net_BatchNorm2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.BatchNorm2d(2)

    def forward(self, input):
        return self.op(input)


_model_ = net_BatchNorm2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
