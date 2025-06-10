import torch


# Generate LeakyReLU operator with Float32, Rank-4 with negative_slope
class net_LeakyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.LeakyReLU(negative_slope=0.5)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_LeakyReLU()

_inputs_ = torch.randn(1, 2, 3, 3)
