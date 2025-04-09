import torch


# Generate LayerNormalization operator with Float32, Rank-4
class net_LayerNormalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.LayerNorm(4)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 17


_model_ = net_LayerNormalization()

_inputs_ = torch.randn(1, 2, 3, 4)
