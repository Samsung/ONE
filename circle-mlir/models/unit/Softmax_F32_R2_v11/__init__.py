import torch


# Generate Softmax operator with Float32, Rank-2, dim=1
class net_Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Softmax(dim=1)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 11


_model_ = net_Softmax()

_inputs_ = torch.randn(3, 5)
