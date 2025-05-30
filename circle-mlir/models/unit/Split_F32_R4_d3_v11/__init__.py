import torch


# Generate Split operator with Float32, Rank-4, dim=3
class net_Split(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.split(input, (1, 2), dim=3)

    def onnx_opset_version(self):
        return 11


_model_ = net_Split()

_inputs_ = torch.randn(2, 2, 3, 3)
