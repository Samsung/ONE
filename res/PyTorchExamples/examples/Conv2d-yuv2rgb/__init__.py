import torch
import torch.nn as nn


# model representing YUVtoRGB conversion
# for details see https://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB
class net_Conv2dYUVtoRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Conv2d(3, 3, 1, bias=False)
        raw_weights = [[1.0, 0.0, 1.13983], \
                       [1.0, -0.39465, -0.58060], \
                       [1.0, 2.03211, 0.0]]
        weights = torch.Tensor(raw_weights).reshape(3, 3, 1, 1)
        self.op.weight = weight = torch.nn.Parameter(weights, requires_grad=False)

    def forward(self, input):
        return torch.clamp(self.op(input), 0.0, 1.0)


_model_ = net_Conv2dYUVtoRGB()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 3, 4, 4)
