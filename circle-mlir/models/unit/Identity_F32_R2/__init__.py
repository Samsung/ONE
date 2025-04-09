import torch


# Generate Identity operator with Float32, Rank-2
class net_Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.id = torch.nn.Identity()

    def forward(self, input):
        return self.id(input)

    def onnx_opset_version(self):
        return 13


_model_ = net_Identity()

_inputs_ = torch.randn(2, 3)
