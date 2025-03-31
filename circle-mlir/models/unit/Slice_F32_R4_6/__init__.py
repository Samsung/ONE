import torch


# Generate Slice operator with Float32, Rank-4,
class net_Slice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input[:, 2:, :, :]

    def onnx_opset_version(self):
        return 14


_model_ = net_Slice()

_inputs_ = torch.randn(1, 4, 1, 2)
