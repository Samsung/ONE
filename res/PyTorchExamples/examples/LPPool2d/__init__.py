import torch
import torch.nn as nn

# model
_model_ = nn.LPPool2d(norm_type=2, kernel_size=2, stride=1)
# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
