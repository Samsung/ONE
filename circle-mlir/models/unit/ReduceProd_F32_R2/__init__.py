import torch


# Generate ReduceProd operator with Float32, Rank-2
class net_ReduceProd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.prod(input)

    def onnx_opset_version(self):
        return 11


_model_ = net_ReduceProd()

_inputs_ = torch.randn(10, 12)
