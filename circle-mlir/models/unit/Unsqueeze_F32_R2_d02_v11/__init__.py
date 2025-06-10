import torch
import onnx


# Generate Unsqueeze operator with Float32, Rank-2 at dim 0, 2
class net_Unsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.unsqueeze(input, 2)

    def onnx_opset_version(self):
        # NOTE onnx-tf fails version >= 13
        return 11

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        for node in onnx_model.graph.node:
            if node.op_type == "Unsqueeze":
                print(node)
                for attribute in node.attribute:
                    print(attribute)
                node.ClearField("attribute")
                new_axes = onnx.helper.make_attribute('axes', [0, 2])
                node.attribute.extend([new_axes])

        # change output shape from [2, 3, 1] to [1, 2, 1, 3]
        for output in onnx_model.graph.output:
            onnx_model.graph.output.remove(output)
        info = onnx.helper.make_tensor_value_info('1', onnx.TensorProto.FLOAT,
                                                  [1, 2, 1, 3])
        onnx_model.graph.output.insert(1, info)
        onnx.save(onnx_model, model_path)


_model_ = net_Unsqueeze()

_inputs_ = torch.randn(2, 3)
