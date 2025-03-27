import torch
import numpy as np


# Generate Equal operator with Int64, Rank-1
class net_Equal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rand = np.random.randint(0, 100, size=(2), dtype=np.int64)
        self.C1 = torch.from_numpy(rand)

    def forward(self, input):
        return torch.eq(input, self.C1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Equal()

rand = np.random.randint(0, 100, size=(2), dtype=np.int64)
_inputs_ = torch.from_numpy(rand)
