import torch
import torch.nn as nn


# model
class net_LPPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.LPPool2d(norm_type=2, kernel_size=1, stride=1)

    def forward(self, input):
        return self.op(input)


_model_ = net_LPPool2d()
# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
