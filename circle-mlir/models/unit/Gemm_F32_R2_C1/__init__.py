import torch


# Generate Gemm operator with Float32, Rank-2, Const
class net_Gemm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Linear(4, 6, bias=True)
        self.input_const = torch.randn(4, 4)

    def forward(self, input):
        return self.op(self.input_const)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 13


_model_ = net_Gemm()

_inputs_ = torch.randn(4, 4)
