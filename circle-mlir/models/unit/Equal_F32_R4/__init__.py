import torch
import numpy as np


# Generate Equal operator with Float32, Rank-4
class net_Equal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.C1 = torch.from_numpy(rng.random((1, 2, 3, 3), dtype=np.float32))

    def forward(self, input):
        return torch.eq(input, self.C1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Equal()

_inputs_ = torch.randn(1, 2, 3, 3)
