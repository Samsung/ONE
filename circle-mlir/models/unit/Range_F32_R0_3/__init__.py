import torch


# Generate Range operator with Float32, scalar
class net_Range(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, start, limit, delta):
        return torch.arange(start, limit, delta, dtype=torch.float32)

    def onnx_opset_version(self):
        return 11


_model_ = net_Range()

# produce float32 scalar with fixed number
_inputs_ = (torch.tensor(0, dtype=torch.float32), torch.tensor(10, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
