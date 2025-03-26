import torch


# Generate Cos operator with Float32, Rank-4
class net_Cos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.cos(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Cos()

_inputs_ = torch.randn(1, 2, 3, 3)
