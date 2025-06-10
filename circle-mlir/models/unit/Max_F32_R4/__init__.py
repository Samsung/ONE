import torch
import numpy as np


# Generate Max operator with Float32, Rank-4
class net_Max(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.mm = torch.from_numpy(rng.random((1, 2, 4, 4), dtype=np.float32))

    def forward(self, input):
        return torch.maximum(input, self.mm)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Max()

_inputs_ = torch.randn(1, 2, 4, 4)
