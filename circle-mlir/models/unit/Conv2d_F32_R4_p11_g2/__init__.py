import torch


# Generate Conv2d operator with Float32, Rank-4 with groups
class net_Conv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Conv2d(4, 4, 1, padding=(1, 1), groups=2)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Conv2d()

_inputs_ = torch.randn(1, 4, 3, 3)
