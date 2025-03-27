import torch


# Generate Neg operator with Float32, Rank-4
class net_Neg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.neg(input)

    def onnx_opset_version(self):
        return 14


_model_ = net_Neg()

_inputs_ = torch.randn(1, 2, 3, 3)
