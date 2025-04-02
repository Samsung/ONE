import torch


# Generate ConvTranspose2d operator with Float32, Rank-4, output_padding (0, 1) stride (1,2)
class net_ConvTranspose2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ConvTranspose2d(4, 6, 1, output_padding=(0, 1), stride=(1, 2))

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_ConvTranspose2d()

_inputs_ = torch.randn(1, 4, 16, 9)
