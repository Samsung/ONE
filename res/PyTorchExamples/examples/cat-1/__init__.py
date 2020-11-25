import torch
import torch.nn as nn


# model
class net_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.cat(inputs, dim=1)


_model_ = net_cat()

# dummy input for onnx generation
_dummy_ = [torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
