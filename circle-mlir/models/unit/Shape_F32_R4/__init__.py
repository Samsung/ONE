import torch


# Generate Shape operator (and Reshape) with Float32, Rank-4
# NOTE this will generate Shape + Reshape network
class net_Shape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1):
        return input0.reshape_as(input1)

    def onnx_opset_version(self):
        return 11


_model_ = net_Shape()

_inputs_ = (torch.randn(1, 2, 4, 4), torch.randn(1, 8, 4))
