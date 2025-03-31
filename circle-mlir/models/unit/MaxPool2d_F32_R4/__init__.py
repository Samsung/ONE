import torch


# Generate MaxPool2d operator with Float32, Rank-4
class net_MaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.MaxPool2d(2)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 12


_model_ = net_MaxPool2d()

_inputs_ = torch.randn(1, 2, 3, 3)
