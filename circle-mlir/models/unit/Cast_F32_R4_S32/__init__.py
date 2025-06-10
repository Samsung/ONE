import torch


# Generate Cast operator with Float32, Rank-4 to Int32
class net_Cast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs.type(torch.int32)

    def onnx_opset_version(self):
        return 14


_model_ = net_Cast()

_inputs_ = torch.randn(1, 2, 3, 3)
