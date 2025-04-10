import torch
import torch.nn.functional as nnf


# Generate Rezize + Conv2D w/padding operator with Float32, Rank-4, nearest mode
# NOTE V11 will produce both scales, sizes attribute exist
class net_Resize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(1, 3, 1, padding=(1, 1))

    def forward(self, input):
        rs = nnf.interpolate(input, size=(16, 9), mode='nearest', align_corners=None)
        c1 = self.conv1(rs)
        return self.conv2(c1)

    def onnx_opset_version(self):
        return 11


_model_ = net_Resize()

_inputs_ = torch.randn(1, 3, 16, 9)
