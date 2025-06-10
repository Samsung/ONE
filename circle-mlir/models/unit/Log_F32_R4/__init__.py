import torch


# Generate Log operator with Float32, Rank-4
class net_Log(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.log(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Log()

_inputs_ = torch.randn(1, 2, 3, 3)
