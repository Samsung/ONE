import torch
import torch.nn as nn


# model
class net_AvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.AvgPool2d(kernel_size=2)

    def forward(self, input):
        return self.op(input)


_model_ = net_AvgPool2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
