import torch
import numpy as np


# Generate Gather operator with Float32, Rank-2, constant data
class net_Gather(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.randn(100, 9)

    def forward(self, index):
        return self.input[index]

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Gather()

# NOTE onnx2circle-value-test's exec_onnx.py generates random int64 into the range [0, 100).
#      Let's use the same range for this test to prevent index values out of data bounds.
rand = np.random.randint(0, 100, size=(3, 9), dtype=np.int64)
_inputs_ = torch.from_numpy(rand)
