import torch
import onnx
import numpy as np


# Generate ConstantOfShape operator with I64, Rank-1
# NOTE replace to ConstantOfShape after creating dummy node
# TODO revise to use pytorch API to create `ConstantOfShape`
class net_ConstantOfShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        C1 = torch.tensor((1, 2), dtype=torch.int64)
        return C1

    def onnx_opset_version(self):
        # TODO set version
        return 9

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        Civ = onnx.helper.make_tensor(name='Ci',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=[1],
                                      vals=[2])
        Ci = onnx.helper.make_node("Constant", [], ['Ci'], value=Civ)

        CAt = onnx.helper.make_tensor(name='At',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=[1],
                                      vals=[1])
        COS = onnx.helper.make_node("ConstantOfShape",
                                    inputs=['Ci'],
                                    outputs=['COS'],
                                    value=CAt,
                                    name='COS')

        onnx_model.graph.node.insert(1, COS)
        onnx_model.graph.node.insert(1, Ci)

        for output in onnx_model.graph.output:
            onnx_model.graph.output.remove(output)

        info = onnx.helper.make_tensor_value_info('COS', onnx.TensorProto.INT64, [2])
        onnx_model.graph.output.insert(1, info)

        onnx.save(onnx_model, model_path)


_model_ = net_ConstantOfShape()

_inputs_ = torch.tensor((1), dtype=torch.int64)
