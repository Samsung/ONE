import torch


# Generate Squeeze operator with Float32, Rank-4, multiple 1 in input, Op version 11
class net_Squeeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.squeeze(input)

    def onnx_opset_version(self):
        return 11


_model_ = net_Squeeze()

_inputs_ = torch.randn(2, 1, 1, 3)
