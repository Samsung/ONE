import torch


# Generate ReduceMean operator with Float32, Rank-4, axis=3
class net_Mean(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.mean(input, axis=3)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Mean()

_inputs_ = torch.randn(1, 2, 4, 4)
