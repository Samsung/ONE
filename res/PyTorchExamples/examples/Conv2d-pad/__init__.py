import torch
import torch.nn as nn


# model
class net_Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Conv2d(1, 1, 1, padding=(1, 0))

    def forward(self, input):
        return self.op(input)


_model_ = net_Conv2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 1, 5, 17)
