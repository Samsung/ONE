import torch
import numpy as np


# Generate Gather operator with Float32, Rank-2, constant int32 type of index
class net_Gather(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rand = np.random.randint(0, 9, size=(3, 9), dtype=np.int32)
        self.index = torch.from_numpy(rand)

    def forward(self, input):
        return input[self.index]

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Gather()

_inputs_ = torch.randn(9, 9)
