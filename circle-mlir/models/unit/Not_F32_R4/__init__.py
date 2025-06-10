import torch


#Generate Not Operator with Float32, Rank-4
class net_Not(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.logical_not(input)

    def onnx_opset_version(self):
        return 14


_model_ = net_Not()

_inputs_ = torch.randn(1, 2, 3, 3)
