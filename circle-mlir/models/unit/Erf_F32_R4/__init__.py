import torch


# Generate Erf operator with Float32, Rank-4
class net_Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.erf(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Erf()

_inputs_ = torch.randn(1, 2, 3, 3)
