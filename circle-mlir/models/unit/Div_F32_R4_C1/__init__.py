import torch
import numpy as np


# Generate Div operator with Float32, Rank-4 with Constant input
class net_div(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.C1 = torch.from_numpy(rng.random((1, 2, 3, 3), dtype=np.float32)) + 1.0

    def forward(self, inputs):
        return torch.div(inputs[0], self.C1)

    def onnx_opset_version(self):
        # TODO set version
        return 14


_model_ = net_div()

_inputs_ = [torch.randn(1, 2, 3, 3)]
