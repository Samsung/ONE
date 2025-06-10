import torch

# Network of  (1N11)- Reshape -(1N1)- Conv -(1M1)- Reshape -(1M11)
# That can be reduced to (1N11) - Conv (1M11)


class net_ReshapeConv1dReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 3, 1)

    def forward(self, input):
        r1 = torch.reshape(input, (1, 4, 1))
        r2 = self.conv(r1)
        return torch.reshape(r2, (1, 3, 1, 1))


_model_ = net_ReshapeConv1dReshape()

_inputs_ = torch.randn(1, 4, 1, 1)


def check_circle_operators(opsDict, operators):
    # Circle.conv2d should exist
    # Circle.reshape should not exist
    if not 'Circle.conv_2d' in operators:
        print('ERROR: Circle.conv_2d NOT exist')
        return 1
    if 'Circle.reshape' in operators:
        print('ERROR: Circle.reshape exist')
        return 1
    return 0
