import torch


# NOTE this is same as Upsample_F32_R4 but with onnx_opset_version 10
class net_Resize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Upsample(scale_factor=(2.0, 2.0))

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # NOTE return 9 produces `Upsample` Op
        return 10


_model_ = net_Resize()

_inputs_ = torch.randn(1, 3, 10, 10)
