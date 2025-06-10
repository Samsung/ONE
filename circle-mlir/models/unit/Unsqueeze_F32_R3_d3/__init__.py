import torch


# Generate Unsqueeze operator with Float32, Rank-3 at dim 3
class net_Unsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.unsqueeze(input, 3)

    def onnx_opset_version(self):
        # NOTE onnx-tf fails version >= 13
        return 14


_model_ = net_Unsqueeze()

_inputs_ = torch.randn(2, 4, 4)
