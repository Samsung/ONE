import torch


# Generate Reciprocal operator with Float32, Rank-4
class net_Reciprocal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.reciprocal(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Reciprocal()

_inputs_ = torch.randn(1, 2, 3, 3)
