import torch
import torch.nn as nn
import numpy as np


# model, equivalent to torch.pixel_unshuffle from torch 1.9+
class net_SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, input):
        # Prepare attributes
        b_size = self.block_size
        batch, input_c, input_h, input_w = list(map(int, list(input.shape)))
        out_c = input_c * b_size * b_size
        out_h = input_h // b_size
        out_w = input_w // b_size

        # Actual model starts here
        x = input.reshape(batch, input_c, out_h, b_size, out_w, b_size)
        x = x.permute([0, 1, 3, 5, 2, 4])
        x = x.reshape([batch, out_c, out_h, out_w])
        return x


_model_ = net_SpaceToDepth(2)

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 6, 6)
