import torch


# Generate ArgMax operator with Float32, Rank-4, keepdims=True, Op-V11
class net_ArgMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.argmax(input, keepdims=True)

    def onnx_opset_version(self):
        return 11


_model_ = net_ArgMax()

_inputs_ = torch.randn(1, 3, 6, 4)
