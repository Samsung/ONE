import torch
import onnx


# Generate Split operator with Float32, Rank-3, no split input
# NOTE It seems not possible to make a Split with only first input.
#      This script will explictly remove second input of Split.
#      This test is to generate odd numbers.
class net_Split(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        q, k, v = torch.split(input, 5, dim=-1)
        return q, k, v

    def onnx_opset_version(self):
        return 14

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        for node in onnx_model.graph.node:
            if node.op_type == "Split":
                print(node)
                if len(node.input) == 2:
                    del node.input[1]
        # delete dangling constant that was 'split'
        for node in onnx_model.graph.node:
            if node.name == "/Constant":
                onnx_model.graph.node.remove(node)

        onnx.save(onnx_model, model_path)


_model_ = net_Split()

_inputs_ = torch.randn(1, 3, 14)
