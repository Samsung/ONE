import torch
import torch.nn as nn


# model
class net_interpolate(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return torch.nn.functional.interpolate(
            input,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True,
            recompute_scale_factor=True)

    def onnx_opset_version(self):
        return 11


_model_ = net_interpolate([2, 2])

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
