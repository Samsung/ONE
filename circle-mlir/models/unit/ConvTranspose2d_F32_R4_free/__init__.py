import torch


# Generate ConvTranspose2d operator with Float32, Rank-4, with all different (I O KH HW)
# Free size test with all different dimension
class net_ConvTranspose2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ConvTranspose2d(3, 5, (1, 2))  # (I O KH KW)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_ConvTranspose2d()

_inputs_ = torch.randn(2, 3, 7, 5)
