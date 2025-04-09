import torch


# Generate GlobalAveragePool operator with Float32, Rank-4
class net_GlobalAveragePool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_GlobalAveragePool()

_inputs_ = torch.randn(1, 2, 3, 9)
