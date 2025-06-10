import torch


# Rewrite test to check FuseConv2DRelu should not work for Conv2D used multiple times
class net_Conv2dReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Conv2d(2, 2, 1)
        self.op2 = torch.nn.ReLU()
        self.op3 = torch.nn.Conv2d(2, 2, 1)

    def forward(self, input):
        r1 = self.op1(input)
        r2 = self.op2(r1)
        r3 = self.op3(r1)
        return torch.sub(r2, r3)


_model_ = net_Conv2dReLU()

_inputs_ = torch.randn(1, 2, 3, 3)


def check_circle_operators(opsDict, operators):
    # Conv2d should exist
    # ReLU should exist for NOT fused
    # Conv2d count should be 2
    # Don't care about Transpose, Sub
    if not 'Circle.conv_2d' in operators:
        print('ERROR: Circle.conv_2d NOT exist')
        return 1
    if not 'Circle.relu' in operators:
        print('ERROR: Circle.relu NOT exist')
        return 1
    if opsDict['Circle.conv_2d'] != 2:
        print('ERROR: Circle.conv_2d NOT 2')
        return 1
    return 0
