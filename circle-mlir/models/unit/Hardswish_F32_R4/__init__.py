import torch


# Generate Hardswish operator with Float32, Rank-4
class net_Hardswish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Hardswish()

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 14


_model_ = net_Hardswish()

_inputs_ = torch.randn(1, 2, 3, 3)
