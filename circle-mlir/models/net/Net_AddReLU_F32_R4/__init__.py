import torch


class net_AddReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.ReLU()
        self.op2 = torch.nn.ReLU()

    def forward(self, input0, input1, input2):
        r1 = torch.add(input0, input1)
        r2 = self.op1(r1)
        r3 = torch.add(r2, input2)
        return self.op2(r3)


_model_ = net_AddReLU()

_inputs_ = (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3))


def check_circle_operators(opsDict, operators):
    # Add should exist
    # Add count should be 2
    # ReLU should not exist
    if not 'Circle.add' in operators:
        print('ERROR: Circle.add NOT exist')
        return 1
    if opsDict['Circle.add'] != 2:
        print('ERROR: Circle.add NOT 2')
        return 1
    if 'Circle.relu' in operators:
        print('ERROR: Circle.relu exist')
        return 1
    return 0
