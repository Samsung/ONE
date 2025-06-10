import torch


# Generate ReduceMax operator with Float32, Rank-4
class net_ReduceMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.max(input)

    def onnx_opset_version(self):
        return 13


_model_ = net_ReduceMax()

_inputs_ = torch.randn(1, 2, 3, 3)
