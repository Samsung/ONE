import torch


# Generate ConvTranspose2d operator with Float32, Rank-4, unknown
# input  : [N, 4, H, 10]
# output : [N, 3, H, 10+7]
# dynamic axes: N, H
class net_ConvTranspose2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ConvTranspose2d(
            in_channels=4,
            out_channels=3,
            kernel_size=(1, 8),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            bias=True,
        )

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_ConvTranspose2d()

_inputs_ = (torch.Tensor(1, 4, 1, 1))

_io_names_ = [['input'], ['output']]
_dynamic_axes_ = {"input": {0: "?", 2: "?"}, "output": {0: "?", 2: "?"}}
