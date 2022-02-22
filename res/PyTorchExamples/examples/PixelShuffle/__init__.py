import torch
import torch.nn as nn


# model
class net_PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.op = torch.nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        return self.op(input)


_model_ = net_PixelShuffle(2)

# dummy input for onnx generation
_dummy_ = torch.randn(1, 8, 3, 3)
