import torch


# Generate BatchNorm2d operator with Float32, Rank-4
# NOTE affine=True(default) will generate mean/var to input, not constant
class net_BatchNorm2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.BatchNorm2d(2, affine=False)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 7


_model_ = net_BatchNorm2d()

_inputs_ = torch.randn(1, 2, 4, 4)
