import torch


# Generate AvgPool2d operator with Float32, Rank-4, padding=[1,1,1,1]
# NOTE this will generate Pad - AveragePool in ONNX
class net_AvgPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.AvgPool2d(2, padding=1)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 7


_model_ = net_AvgPool2d()

_inputs_ = torch.randn(1, 2, 4, 4)
