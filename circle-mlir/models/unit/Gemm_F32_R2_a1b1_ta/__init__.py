import torch
import onnx
import numpy as np


# Generate Gemm operator with Float32, Rank-2
# Gemm with alpha 1.0, beta 1.0, transA=1, transB=0
# - there is no direct interface to set transA/transB
# - create input with Identity to output as transpose of original input
# - and use that transposed input as Gemm Op with 'transA=1'
class net_Gemm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat2 = torch.randn(4, 5)
        self.bias = torch.randn(3, 5)

    def forward(self, inputs):
        beta = 1.0
        alpha = 1.0
        # torch.addmm(input, mat1i, mat2i, *, beta=1, alpha=1, out=None)
        # out=βinput + α(mat1i ​@ mat2i​)
        return torch.addmm(self.bias, inputs[0], self.mat2, beta=beta,
                           alpha=alpha), inputs[1]

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 8

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # NOTE used fixed input/output index from generation result and may differ in the future
        for node in onnx_model.graph.node:
            if node.op_type == "Gemm":
                new_transA = onnx.helper.make_attribute('transA', 1)
                node.attribute.extend([new_transA])
                node.input[0] = "onnx::Identity_1"
                print(node)
            if node.op_type == "Identity":
                # remove dummy Identity
                onnx_model.graph.node.remove(node)

        # remove graph I/O for Identity Op
        onnx_model.graph.output.remove(onnx_model.graph.output[1])
        onnx_model.graph.input.remove(onnx_model.graph.input[0])

        onnx.save(onnx_model, model_path)


_model_ = net_Gemm()

_inputs_ = [torch.randn(3, 4), torch.randn(4, 3)]
