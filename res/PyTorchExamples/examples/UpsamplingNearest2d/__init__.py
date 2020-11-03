import torch
import torch.nn as nn


# model
class net_UpsamplingNearest2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.UpsamplingNearest2d(2)

    def forward(self, input):
        return self.op(input)


_model_ = net_UpsamplingNearest2d()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 4, 4)
