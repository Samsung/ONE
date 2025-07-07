import torch


# Generate Conv2d operator with Float32, Rank-4 with groups.
# this graph is not convertable to DepthwiseConv and should be
# converted to Transpose-Split-Conv-Concat-Add-Transpose.
class net_Conv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Conv2d(8, 4, 1, groups=4)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14


_model_ = net_Conv2d()

_inputs_ = torch.randn(1, 8, 6, 3)


def check_circle_operators(opsDict, operators):
    # Conv2d count should be 4
    # Split count should be 1
    # Concatenation count should be 1
    # Add count should be 1
    if not 'Circle.conv_2d' in operators:
        print('ERROR: Circle.conv_2d NOT exist')
        return 1
    if not 'Circle.split' in operators:
        print('ERROR: Circle.split NOT exist')
        return 1
    if not 'Circle.concatenation' in operators:
        print('ERROR: Circle.concatenation NOT exist')
        return 1
    if not 'Circle.add' in operators:
        print('ERROR: Circle.add NOT exist')
        return 1
    if opsDict['Circle.conv_2d'] != 4:
        print('ERROR: Circle.conv_2d NOT 4')
        return 1
    if opsDict['Circle.split'] != 1:
        print('ERROR: Circle.split NOT 1')
        return 1
    if opsDict['Circle.concatenation'] != 1:
        print('ERROR: Circle.concatenation NOT 1')
        return 1
    if opsDict['Circle.add'] != 1:
        print('ERROR: Circle.add NOT 1')
        return 1
    return 0
