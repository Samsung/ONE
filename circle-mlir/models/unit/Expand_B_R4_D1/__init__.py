import torch


# Generate Expand operator with bool, Rank-4 at D1
class net_Expand(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.expand(-1, 8, -1, -1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 9


_model_ = net_Expand()

_inputs_ = torch.randint(1, (2, 1, 3, 4), dtype=torch.bool)
