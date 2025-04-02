import torch


# Generate ReduceSum operator with Float32, Rank-2, dim=1, keepdim=True
class net_ReduceSum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(input, 1, keepdim=True)

    def onnx_opset_version(self):
        return 13


_model_ = net_ReduceSum()

_inputs_ = torch.randn(10, 12)
