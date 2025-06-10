import torch


# Generate Unsqueeze operator with Float32, scalar
class net_Unsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.unsqueeze(input, 0)

    def onnx_opset_version(self):
        # NOTE onnx-tf fails version >= 13
        return 14


_model_ = net_Unsqueeze()

# produuce float32 scalar
_inputs_ = torch.randn(1)[0]
