import torch


# Generate MaxPool2d operator with Float32, Rank-4, stride=[2,2], padding=[3,1,3,1]
class net_MaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.MaxPool2d(6, stride=2, padding=(3, 1))

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # use 9 for no 'ceil_mode'
        return 9


_model_ = net_MaxPool2d()

_inputs_ = torch.randn(1, 2, 16, 16)
