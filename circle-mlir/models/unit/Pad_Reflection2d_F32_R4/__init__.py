import torch


# Generate Pad / reflect mode operator with Float32, Rank-4
# NOTE "torch.reshape" is added at the end to make model's output dim all known.
# subgraph generated for paddings of "Pad" will endup with unknown dims.
# when conversion and folding is done, there will be mismatch of shapes of
# output of Circle.MirrorPad and the model.
class net_ReflectionPad2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.ReflectionPad2d((0, 0, 1, 1))

    def forward(self, input):
        pad = self.op(input)
        return torch.reshape(pad, (1, 2, 5, 3))

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 11


_model_ = net_ReflectionPad2d()

_inputs_ = torch.randn(1, 2, 3, 3)
