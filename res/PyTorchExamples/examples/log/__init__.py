import torch
import torch.nn as nn


# model
class net_log(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.log(input)


_model_ = net_log()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
