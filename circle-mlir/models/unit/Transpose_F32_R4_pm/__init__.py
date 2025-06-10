import torch


# Generate Transpose operator with Float32, Rank-4, with permute()
class net_Transpose(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # NOTE produce NCHW to NHWC
        return input.permute(0, 2, 3, 1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Transpose()

_inputs_ = torch.randn(1, 3, 6, 4)
