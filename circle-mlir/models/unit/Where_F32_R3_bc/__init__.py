import torch


# Generate Where operator with F32, Rank-3, broadcasting
class net_Where(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1, input2):
        return torch.where(input0, input1, input2)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Where()

ints = torch.randn(2, 1, 3)
bools = torch.gt(ints, 0.5)

_inputs_ = (bools, torch.randn(1), torch.randn(1, 2, 3))
