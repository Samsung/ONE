import torch


# Generate LogSoftmax operator with Float32, Rank-2
class net_LogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_LogSoftmax()

_inputs_ = torch.randn(3, 5)
