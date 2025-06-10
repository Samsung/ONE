import torch


# Generate ReduceSum operator with Float32, Rank-4
class net_ReduceSum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(input)

    def onnx_opset_version(self):
        return 13


_model_ = net_ReduceSum()

_inputs_ = torch.randn(1, 2, 3, 3)
