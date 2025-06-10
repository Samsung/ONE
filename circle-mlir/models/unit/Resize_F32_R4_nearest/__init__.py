import torch
import numpy as np


# Generate Resize with Float32, Rank-4
# mode=nearest, coordinate_transformation_mode=asymmetric
# NOTE "torch.add" is added at the end to make model's output dim all known.
# subgraph generated for size of "Resize" will endup with unknown dims.
# when conversion and folding is done, there will be mismatch of shapes of
# output of Circle.ResizeNearestNeighbor and the model.
class net_Resize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.C1 = torch.from_numpy(rng.random((2, 3, 8, 6), dtype=np.float32))
        self.op = torch.nn.Upsample(scale_factor=(2, 2),
                                    mode="nearest",
                                    recompute_scale_factor=False)

    def forward(self, input):
        upsample = self.op(input)
        # NOTE 'add' is added to make model output shape as all known
        return torch.add(upsample, self.C1)

    def onnx_opset_version(self):
        return 11


_model_ = net_Resize()

_inputs_ = torch.randn(2, 3, 4, 3)
