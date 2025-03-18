import torch


# Generate Reshape operator with Float32, Rank-4
class net_Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.reshape(input, (1, 1, 1, 8))

    def onnx_opset_version(self):
        return 14


_model_ = net_Reshape()

_inputs_ = torch.randn(1, 1, 2, 4)
