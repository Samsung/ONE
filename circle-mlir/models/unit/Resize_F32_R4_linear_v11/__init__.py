import torch


# Generate Rezize operator with Float32, Rank-4, linear mode
# NOTE mode is 'bilinear' but actual Op will have mode='linear', nearest_mode='floor'
# NOTE V11 will produce both scales, sizes attribute exist
class net_Resize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Upsample(size=(32, 18), mode='bilinear', align_corners=True)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 11


_model_ = net_Resize()

_inputs_ = torch.randn(1, 3, 16, 9)
