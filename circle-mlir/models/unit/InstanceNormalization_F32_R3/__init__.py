import torch


# Generate InstanceNormalization operator with Float32, Rank-3
class net_InstanceNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.GroupNorm(6, 6)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 7


_model_ = net_InstanceNorm()

_inputs_ = torch.randn(1, 6, 2)
