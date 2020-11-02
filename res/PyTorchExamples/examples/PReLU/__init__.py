import torch
import torch.nn as nn


# model
class net_PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.PReLU()

    def forward(self, input):
        return self.op(input)


_model_ = net_PReLU()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
