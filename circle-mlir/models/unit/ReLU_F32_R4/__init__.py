import torch


# Generate ReLU operator with Float32, Rank-4
class net_ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ReLU()

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_ReLU()

_inputs_ = torch.randn(1, 2, 3, 3)
