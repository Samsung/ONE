import torch


# Generate Resize with Float32, Rank-4
# mode=nearest, coordinate_transformation_mode=asymmetric
# with dynamic input shape
class net_Resize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Upsample(scale_factor=(2.0, 2.0),
                                    mode="nearest",
                                    recompute_scale_factor=False)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 13


_model_ = net_Resize()

_inputs_ = torch.randn(2, 3, 4, 3)

_io_names_ = [['input'], ['output']]
_dynamic_axes_ = {'input': {0: '?'}}
