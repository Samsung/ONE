import torch
import numpy as np


# Generate MatMul operator with Float32, Rank-4 and Constant input (11hw)
class net_MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.B = torch.from_numpy(rng.random((1, 1, 4, 6), dtype=np.float32))

    def forward(self, inputs):
        return torch.matmul(inputs[0], self.B)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_MatMul()

_inputs_ = [torch.randn(1, 1, 1, 4)]
