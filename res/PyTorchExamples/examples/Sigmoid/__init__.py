import torch
import torch.nn as nn


# model
class net_Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Sigmoid()

    def forward(self, input):
        return self.op(input)


_model_ = net_Sigmoid()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
