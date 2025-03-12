import torch


# Generate Pad / constant mode operator with Float32, Rank-4, ver 14
class net_ConstantPad2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ConstantPad2d(0, 0.0)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 14


_model_ = net_ConstantPad2d()

_inputs_ = torch.randn(1, 2, 3, 3)
