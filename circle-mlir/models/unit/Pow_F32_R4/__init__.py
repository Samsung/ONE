import torch


# Generate Pow operator with Float32, Rank-4
class net_Pow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.pow(input, 2.0)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 7


_model_ = net_Pow()

_inputs_ = torch.randn(1, 2, 3, 3)
