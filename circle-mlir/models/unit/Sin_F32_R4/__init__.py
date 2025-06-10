import torch


# Generate Sin operator with Float32, Rank-4
class net_Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Sin()

_inputs_ = torch.randn(1, 2, 3, 3)
