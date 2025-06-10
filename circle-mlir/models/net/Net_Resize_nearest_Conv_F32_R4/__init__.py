import torch
import torch.nn.functional as nnf


# Generate Rezize + Conv2D operator with Float32, Rank-4, nearest mode
# NOTE V11 will produce both scales, sizes attribute exist
class net_Resize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, input):
        rs = nnf.interpolate(input, size=(16, 9), mode='nearest', align_corners=None)
        return self.conv(rs)

    def onnx_opset_version(self):
        return 11


_model_ = net_Resize()

_inputs_ = torch.randn(1, 3, 16, 9)


def check_circle_operators(opsDict, operators):
    # ResizeNearestNeighbor should exist
    # Slice should not exist
    # Concat should not exist
    if 'Circle.concatenation' in operators:
        print('ERROR: Circle.concatenation exist')
        return 1
    if 'Circle.slice' in operators:
        print('ERROR: Circle.slice exist')
        return 1
    if not 'Circle.resize_nearest_neighbor' in operators:
        print('ERROR: Circle.resize_nearest_neighbor NOT exist')
        return 1
    return 0
