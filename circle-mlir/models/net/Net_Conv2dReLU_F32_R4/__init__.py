import torch


class net_Conv2dReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Conv2d(2, 2, 1)
        self.op2 = torch.nn.ReLU()
        self.op3 = torch.nn.Conv2d(2, 2, 1)
        self.op4 = torch.nn.ReLU()

    def forward(self, input):
        r1 = self.op1(input)
        r2 = self.op2(r1)
        r3 = self.op3(r2)
        return self.op4(r3)


_model_ = net_Conv2dReLU()

_inputs_ = torch.randn(1, 2, 3, 3)


def check_circle_operators(opsDict, operators):
    # Conv2d and Transpose should exist
    # Conv2d count should be 2
    # ReLU should not exist
    if not 'Circle.conv_2d' in operators:
        print('ERROR: Circle.conv_2d NOT exist')
        return 1
    if not 'Circle.transpose' in operators:
        print('ERROR: Circle.transpose NOT exist')
        return 1
    if opsDict['Circle.conv_2d'] != 2:
        print('ERROR: Circle.conv_2d NOT 2')
        return 1
    if 'Circle.relu' in operators:
        print('ERROR: Circle.relu exist')
        return 1
    return 0
