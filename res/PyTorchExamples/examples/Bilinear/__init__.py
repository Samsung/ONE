import torch
import torch.nn as nn


# model
class net_Bilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Bilinear(20, 30, 40)

    def forward(self, inputs):
        return self.op(inputs[0], inputs[1])


_model_ = net_Bilinear()

# dummy input for onnx generation
_dummy_ = [torch.randn(128, 20), torch.randn(128, 30)]
