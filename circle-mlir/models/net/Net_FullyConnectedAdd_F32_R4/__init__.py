import torch
import numpy as np


class net_FullyConnectedAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.op = torch.nn.Linear(4, 6, bias=True)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        return 13


_model_ = net_FullyConnectedAdd()

_inputs_ = torch.randn(1, 1, 3, 4)


def check_circle_operators(opsDict, operators):
    # FullyConnected should exist
    # Add should not exist
    if not 'Circle.fully_connected' in operators:
        print('ERROR: Circle.fully_connected NOT exist')
        return 1
    if 'Circle.add' in operators:
        print('ERROR: Circle.add exist')
        return 1
    return 0
