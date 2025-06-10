import torch


# NOTE this is same as Resize_F32_R4 but with onnx_opset_version 9
# NOTE Upsample Op is now deprecated, last version is 9
class net_Upsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Upsample(scale_factor=(2.0, 2.0))

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # NOTE return 10 produces `Resize` Op
        return 9


_model_ = net_Upsample()

_inputs_ = torch.randn(1, 3, 16, 9)
