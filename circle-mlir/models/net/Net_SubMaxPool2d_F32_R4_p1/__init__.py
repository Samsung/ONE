import torch


# Generate Sub-MaxPool2d network with Float32, Rank-4, padding=[1,1,1,1]
# Regression test to feed minus values to MaxPool2d
class net_SubMaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.MaxPool2d(3, padding=1)

    def forward(self, input):
        vs = torch.sub(input, 100.0)
        return self.op(vs)

    def onnx_opset_version(self):
        # use 9 for no 'ceil_mode'
        return 9


_model_ = net_SubMaxPool2d()

_inputs_ = torch.randn(1, 2, 7, 7)
