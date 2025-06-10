import torch


# Generate Sigmoid operator with Float32, Rank-4
class net_Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Sigmoid()

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Sigmoid()

_inputs_ = torch.randn(1, 2, 3, 3)
