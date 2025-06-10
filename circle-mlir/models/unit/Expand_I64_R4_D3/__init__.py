import torch


# Generate Expand operator with Int64, Rank-4 at D3
class net_Expand(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.expand(-1, -1, 4, -1)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 9


_model_ = net_Expand()

_inputs_ = torch.randint(low=-16, high=16, size=(1, 1, 1, 4), dtype=torch.int64)
