import torch


# Generate Slice operator with Float32, Rank-4, minus values
class net_Slice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input[:, :, :, -8:-4]

    def onnx_opset_version(self):
        return 14


_model_ = net_Slice()

_inputs_ = torch.randn(2, 2, 2, 16)
