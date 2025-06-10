import torch
import onnx


# Generate Gemm operator with Float32, Rank-2, no bias
# Gemm with alpha 1.0, beta 1.0, transA=0, transB=1
class net_Gemm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat2 = torch.randn(4, 5)
        self.bias = torch.randn(3, 5)

    def forward(self, input):
        beta = 1.0
        alpha = 1.0
        # torch.addmm(input, mat1i, mat2i, *, beta=1, alpha=1, out=None)
        # out=βinput + α(mat1i ​@ mat2i​)
        return torch.addmm(self.bias, input, self.mat2, beta=beta, alpha=alpha)

    def onnx_opset_version(self):
        return 11

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        for node in onnx_model.graph.node:
            print(node)
            if node.op_type == "Gemm":
                constant_del = node.input[2]
                del node.input[2]
                break

        # remove bias constant from graph
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] == constant_del:
                    onnx_model.graph.node.remove(node)
                    break

        onnx.save(onnx_model, model_path)


_model_ = net_Gemm()

_inputs_ = torch.randn(3, 4)
