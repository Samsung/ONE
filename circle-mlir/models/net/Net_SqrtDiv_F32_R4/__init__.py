import torch
import numpy as np


# Generate Sqrt-Div operator with Float32, Rank-4
class net_SqrtDiv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = torch.from_numpy(np.ones((1, 2, 3, 3), dtype=np.float32))

    def forward(self, input):
        r1 = torch.sqrt(input)
        return torch.div(self.C1, r1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_SqrtDiv()

_inputs_ = torch.randn(1, 2, 3, 3)


def check_circle_operators(opsDict, operators):
    # Check Rewrite for Div(1.0, Sqrt(X)) -> Rsqrt(X)
    # Div, Sqrt should not exist
    # Rsqrt should exist
    if 'Circle.div' in operators:
        print('ERROR: Circle.div exist')
        return 1
    if 'Circle.sqrt' in operators:
        print('ERROR: Circle.sqrt exist')
        return 1
    if not 'Circle.rsqrt' in operators:
        print('ERROR: Circle.rsqrt NOT exist')
        return 1
    return 0
