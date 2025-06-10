import torch


# Generate Gemm operator with Float32, Rank-2
# Gemm with alpha 0.5, beta 0.5, transA=0, transB=1
class net_Gemm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat2 = torch.randn(4, 5)
        self.bias = torch.randn(3, 5)

    def forward(self, input):
        beta = 0.5
        alpha = 0.5
        # torch.addmm(input, mat1i, mat2i, *, beta=1, alpha=1, out=None)
        # out=βinput + α(mat1i ​@ mat2i​)
        return torch.addmm(self.bias, input, self.mat2, beta=beta, alpha=alpha)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 8


_model_ = net_Gemm()

_inputs_ = torch.randn(3, 4)
