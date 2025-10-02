import torch
import numpy as np


# Generate Gather operator with Float32, Rank-2, no constant inputs
class net_Gather(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1):
        return input0[input1]

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Gather()

# NOTE onnx2circle-value-test's exec_onnx.py generates random int64 into the range [0, 100).
#      Let's use the same range for this test to prevent index values out of data bounds.
rand = np.random.randint(0, 100, size=(3, 9), dtype=np.int64)
_inputs_ = (torch.randn(100, 9), torch.from_numpy(rand))
