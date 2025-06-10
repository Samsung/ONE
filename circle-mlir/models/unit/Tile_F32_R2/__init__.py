import torch
import onnx


# Generate Tile operator with Float32, Rank-2
class net_Tile(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.tile(input, (2, 2))

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # delete all nodes except "Tile" and input "Constant"
        toRemove = ['Shape', 'Size', 'Sub', 'Greater', 'Less', 'If', 'Cast']
        contRemove = True
        didRemove = False
        while contRemove:
            didRemove = False
            for node in onnx_model.graph.node:
                if node.op_type == 'Constant':
                    # NOTE skip removing constant as node.name is different
                    #      across different onnx/onnxruntime.
                    #      onnxrt 1.16.0 checked for "node.name != 'Constant_0'"
                    #      but 1.18.0 gives 'Constant_4'
                    break
                elif node.op_type in toRemove:
                    onnx_model.graph.node.remove(node)
                    didRemove = True
                    break
            if not didRemove:
                contRemove = False

        for node in onnx_model.graph.node:
            if node.op_type == "Tile":
                node.input[0] = "onnx::Shape_0"
                node.input[1] = "1"  # output of "Constant_0" is "1"

        onnx.save(onnx_model, model_path)


_model_ = net_Tile()

_inputs_ = torch.randn(2, 3)
