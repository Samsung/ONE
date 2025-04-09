import torch


# Generate Flattten operator with Float32, Rank-3 to Rank-2
class net_Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.flatten(input, start_dim=1, end_dim=2)

    def onnx_opset_version(self):
        return 14


_model_ = net_Reshape()

_inputs_ = torch.randn(2, 3, 8)
