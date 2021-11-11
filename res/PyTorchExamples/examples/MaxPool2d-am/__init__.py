import torch
import torch.nn as nn


# model
class net_MaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.MaxPool2d(3, stride=1, return_indices=True)

    def forward(self, input):
        return self.op(input)


_model_ = net_MaxPool2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 4, 4)
