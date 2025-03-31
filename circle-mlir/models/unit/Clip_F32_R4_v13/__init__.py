import torch


# Generate Clip operator with Float32, Rank-4
class net_Clip(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.clip(input, 0.0, 6.0)

    def onnx_opset_version(self):
        return 14


_model_ = net_Clip()

_inputs_ = torch.randn(1, 2, 3, 3)
