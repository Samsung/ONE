import torch


# Generate MatMul operator with Float32, Rank-3(NHW)
# Filter with Rank-3 for regression test
class net_MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.matmul(inputs[0], inputs[1])

    def onnx_opset_version(self):
        return 14


_model_ = net_MatMul()

_inputs_ = [torch.randn(4, 1, 8), torch.randn(4, 8, 1)]
