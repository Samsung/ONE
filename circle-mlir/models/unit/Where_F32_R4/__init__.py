import torch


# Generate Where operator with F32, Rank-4
class net_Where(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.where(input[0], input[1], input[2])

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Where()

ints = torch.randn(1, 2, 3, 3)
bools = torch.gt(ints, 0.5)

_inputs_ = [bools, torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
