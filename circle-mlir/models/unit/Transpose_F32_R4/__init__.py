import torch


# Generate Transpose operator with Float32, Rank-4
class net_Transpose(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # NOTE two transpose will produce NCHW to NHWC single transpose
        x = torch.transpose(input, 1, 3)
        return torch.transpose(x, 1, 2)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Transpose()

_inputs_ = torch.randn(1, 3, 6, 4)
