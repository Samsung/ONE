import torch
import torch.nn as nn


# model
class net_Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = nn.Conv2d(32, 2, 1)
        self.op2 = nn.Conv2d(32, 16, 1)
        self.op3 = nn.Conv2d(32, 32, 1)

    def forward(self, input):
        return self.op1(input), self.op2(input), self.op3(input)


_model_ = net_Conv2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 32, 3, 3)
