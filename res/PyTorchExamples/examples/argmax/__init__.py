import torch
import torch.nn as nn


# model
class net_argmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.argmax(input)


_model_ = net_argmax()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
