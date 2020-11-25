import torch
import torch.nn as nn


# model
class net_ReLU6(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU6()

    def forward(self, input):
        return self.op(input)


_model_ = net_ReLU6()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
