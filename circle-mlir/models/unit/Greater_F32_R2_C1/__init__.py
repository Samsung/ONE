import torch


# Generate Greater operator with Float32, Rank-2, Constant scalar data
class net_Greater(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.greater(input, 0)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Greater()

_inputs_ = torch.randn(3, 4)
