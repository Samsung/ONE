import torch
import torch.nn as nn


# model
#
# Notes:
# - This model requires opset version 10+. Previous version does not support strides.
class net_strided_slice(nn.Module):
    def __init__(self, begin, end, stride):
        super().__init__()
        self.key = [slice(begin[i], end[i], stride[i]) for i in range(len(begin))]

    def forward(self, input):
        # this is general way to do input[:, :, 1:5:2, 0:5:2]
        return input[self.key]

    def onnx_opset_version(self):
        return 10


_model_ = net_strided_slice([0, 0, 1, 0], [1, 3, 5, 5], [1, 1, 2, 2])

# dummy input for onnx generation
_dummy_ = torch.randn(1, 3, 5, 5)
