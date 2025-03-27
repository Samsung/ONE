import torch


# Generate CumSum operator with Float32, Rank-4
class net_CumSum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # Axis = 1, identical tensor is expected.
        return torch.cumsum(input, 1)

    def onnx_opset_version(self):
        return 14


_model_ = net_CumSum()

_inputs_ = torch.randn(1, 1, 10, 10)
