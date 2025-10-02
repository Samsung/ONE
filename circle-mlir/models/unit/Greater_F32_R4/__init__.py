import torch


# Generate Greater operator with Float32, Rank-4
class net_Greater(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1):
        return torch.greater(input0, input1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Greater()

_inputs_ = (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3))
