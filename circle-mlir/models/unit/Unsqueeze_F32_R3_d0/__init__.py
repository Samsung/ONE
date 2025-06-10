import torch


# Generate Unsqueeze operator with Float32, Rank-3 at dim 0
class net_Unsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.unsqueeze(input, 0)

    def onnx_opset_version(self):
        # NOTE onnx-tf fails version >= 13
        return 14


_model_ = net_Unsqueeze()

_inputs_ = torch.randn(3, 4, 4)
