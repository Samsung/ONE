import torch


# Generate Split operator with Float32, Rank-4
class net_Split(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.split(input, (3, 4, 2), dim=1)

    def onnx_opset_version(self):
        return 14


_model_ = net_Split()

_inputs_ = torch.randn(2, 9, 3, 3)
