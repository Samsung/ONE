import torch


# Rewrite test to check Concat is removed if there is only one input
class net_AddReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.ReLU()

    def forward(self, inputs):
        r1 = self.op1(inputs)
        r2 = torch.cat([r1], dim=2)
        r3 = torch.add(r2, inputs)
        return torch.cat([r3], dim=2)


_model_ = net_AddReLU()

_inputs_ = torch.randn(1, 2, 3, 3)


def check_circle_operators(opsDict, operators):
    # ReLU, Add should exist
    # Concat should not exist
    if not 'Circle.add' in operators:
        print('ERROR: Circle.add NOT exist')
        return 1
    if not 'Circle.relu' in operators:
        print('ERROR: Circle.relu NOT exist')
        return 1
    if 'Circle.concatenation' in operators:
        print('ERROR: Circle.concatenation exist')
        return 1
    return 0
