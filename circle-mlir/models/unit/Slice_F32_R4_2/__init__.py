import torch


# Generate Slice operator with Float32, Rank-4
class net_Slice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input[:, :, :, 4:8]

    def onnx_opset_version(self):
        return 14


_model_ = net_Slice()

_inputs_ = torch.randn(1, 1, 1, 16)
