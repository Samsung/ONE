import torch


# Generate Rsqrt operator with Float32, Rank-4
class net_rsqrt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.rsqrt(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_rsqrt()

_inputs_ = torch.randn(1, 2, 3, 3)
