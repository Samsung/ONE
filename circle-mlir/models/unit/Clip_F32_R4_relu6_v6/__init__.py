import torch


# Generate Clip operator with Float32, Rank-4
class net_Clip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ReLU6()

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 7


_model_ = net_Clip()

_inputs_ = torch.randn(1, 2, 3, 3)
