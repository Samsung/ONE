import torch


# Generate ReduceMax operator with Float32, Rank-2, dim=1
class net_ReduceMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.max(input, 1)

    def onnx_opset_version(self):
        return 13


_model_ = net_ReduceMax()

_inputs_ = torch.randn(10, 12)
