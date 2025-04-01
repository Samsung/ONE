import torch


# Generate Expand operator with Float32, Rank-4
class net_Expand(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.expand(2, 3, 4, 5)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 9


_model_ = net_Expand()

_inputs_ = torch.randn(1)
