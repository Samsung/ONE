import torch


# Generate Floor operator with Float32, Rank-4
class net_Floor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.floor(input)

    def onnx_opset_version(self):
        return 11


_model_ = net_Floor()

_inputs_ = torch.randn(1, 2, 3, 3)
