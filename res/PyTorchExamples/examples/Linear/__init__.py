import torch
import torch.nn as nn


# model
class net_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Linear(3, 3)

    def forward(self, input):
        return self.op(input)


_model_ = net_Linear()
# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
