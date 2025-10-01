import torch


# Generate Reshape operator with Float32, Rank-4 to Rank-3
class net_Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1):
        a = input0.reshape_as(input1)
        b = torch.reshape(a, (1, 8, -1))
        c = torch.transpose(b, 1, 2)
        d = c + 1
        return d

    def onnx_opset_version(self):
        return 14


_model_ = net_Reshape()

_inputs_ = torch.randn(1, 3, 3, 8), torch.randn(1, 8, 3, 3)
