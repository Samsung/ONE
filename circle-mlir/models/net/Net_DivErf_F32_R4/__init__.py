import torch
import numpy as np


# Generate Div-Erf operator with Float32, Rank-4
class net_DivErf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.C1 = torch.from_numpy(rng.random((1, 2, 3, 3), dtype=np.float32)) + 1.0

    def forward(self, input):
        r1 = torch.div(input, self.C1)
        return torch.erf(r1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_DivErf()

_inputs_ = torch.randn(1, 2, 3, 3)


def check_circle_operators(opsDict, operators):
    # Div should not exist
    # Mul should exist
    if 'Circle.div' in operators:
        print('ERROR: Circle.div exist')
        return 1
    if not 'Circle.mul' in operators:
        print('ERROR: Circle.mul NOT exist')
        return 1
    return 0
