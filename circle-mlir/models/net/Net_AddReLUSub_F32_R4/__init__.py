import torch


# Rewrite test to check FuseAddRelu should not work for Add used multiple times
class net_AddReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.ReLU()

    def forward(self, input):
        r1 = torch.add(input, input)
        r2 = self.op1(r1)
        return torch.sub(r1, r2)


_model_ = net_AddReLU()

_inputs_ = torch.randn(1, 2, 3, 3)


def check_circle_operators(opsDict, operators):
    # Add should exist
    # ReLU should exist for NOT fused
    # Add count should be 1
    # Don't care about Sub
    if not 'Circle.add' in operators:
        print('ERROR: Circle.add NOT exist')
        return 1
    if not 'Circle.relu' in operators:
        print('ERROR: Circle.relu NOT exist')
        return 1
    if opsDict['Circle.add'] != 1:
        print('ERROR: Circle.add NOT 1')
        return 1
    return 0
