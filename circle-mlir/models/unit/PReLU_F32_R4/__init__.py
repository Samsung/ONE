import torch


# Generate PReLU operator with Float32, Rank-4
class net_PReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.PReLU()

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_PReLU()

_inputs_ = torch.randn(1, 2, 3, 3)
