import torch
import onnx
import numpy as np


# Generate Gemm operator with Float32, Rank-2
# Gemm with alpha 1.0, beta 1.0, transA=0, transB=1
# - there is no direct interface to set transA/transB
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
        # TODO set to appropriate value
        return 8

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # make new mat2 with transposed shape
        new_b_shape = [5, 4]
        new_b_name = '/Constant_output_0_t'
        new_b_v = onnx.helper.make_tensor(name=new_b_name,
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=new_b_shape,
                                          vals=np.random.randn(5 * 4).flatten().astype(
                                              np.float32))

        new_b = onnx.helper.make_node("Constant", [], [new_b_name],
                                      name='/Constant_0_t',
                                      value=new_b_v)

        onnx_model.graph.node.insert(0, new_b)

        for node in onnx_model.graph.node:
            if node.op_type == "Gemm":
                new_transB = onnx.helper.make_attribute('transB', 1)
                node.attribute.extend([new_transB])
                node.input[1] = new_b_name
                print(node)

        onnx.save(onnx_model, model_path)


_model_ = net_Gemm()

_inputs_ = torch.randn(3, 4)
