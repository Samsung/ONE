import torch


# Generate Conv2d operator with Float32, Rank-4, stride=2
class net_Conv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Conv2d(2, 2, kernel_size=(1, 1), stride=2)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 12


_model_ = net_Conv2d()

_inputs_ = torch.randn(1, 2, 5, 5)
