import torch
import torch.nn as nn


# model
class net_Conv2dPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = nn.Conv2d(1, 1, 1)
        self.op2 = nn.PReLU()

    def forward(self, input):
        return self.op2(self.op1(input))


_model_ = net_Conv2dPReLU()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 1, 5, 5)
