import torch


# Generate Reshape operator with Float32, Rank-4 to Rank-3
class net_Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        a = input[0].reshape_as(input[1])
        b = torch.reshape(a, (1, 8, -1))
        c = torch.transpose(b, 1, 2)
        d = torch.cat((c, c), 1)
        e = d + 1
        return e

    def onnx_opset_version(self):
        return 14


_model_ = net_Reshape()

_inputs_ = [torch.randn(1, 3, 3, 8), torch.randn(1, 8, 3, 3)]


def check_circle_operators(opsDict, operators):
    # TODO Need to implement checking shapes of operators
    # This model is to check whether following operators are statically and correctly inferred
    # Currently, we don't have such verifying logic, so it just returns success(0).
    return 0
