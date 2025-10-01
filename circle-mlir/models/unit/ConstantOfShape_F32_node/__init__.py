import torch
import onnx
import numpy as np


# Generate ConstantOfShape operator with F32, input is from non-constant node
class net_ConstantOfShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(seed=123)
        self.C1 = torch.from_numpy(rng.random((1, 8, 4), dtype=np.float32))

    def forward(self, input0, input1):
        res = input0.reshape_as(input1)
        return torch.add(res, self.C1)

    def onnx_opset_version(self):
        # TODO set version
        return 9

    def post_process(self, model_path):
        '''
        Before
            [1x2x4x4] [1x8x4]
                 |       |
                 |     Shape
                  \     /
                   \   /
                 Reshape Constant
                     |   /
                      \ /
                      Add
                       |
                    [1x8x4]

        After
            [1x2x4x4] [1x8x4]
                 |       |
                 |     Shape
                  \     / \
                   \   /   |
                 Reshape ConstantOfShape
                     |    /
                      \  /
                      Add
                       |
                    [1x8x4]
        '''

        onnx_model = onnx.load(model_path)

        CAt = onnx.helper.make_tensor(name='const_value',
                                      data_type=onnx.TensorProto.FLOAT,
                                      dims=[1],
                                      vals=[2.718])
        COS = onnx.helper.make_node("ConstantOfShape",
                                    inputs=["/Shape_output_0"],
                                    outputs=["ConstOut"],
                                    value=CAt,
                                    name='constant')
        onnx_model.graph.node.insert(1, COS)

        for node in onnx_model.graph.node:
            if node.op_type == "Add":
                node.input[1] = "ConstOut"

        # remove Constant from graph
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant':
                onnx_model.graph.node.remove(node)
                break

        onnx.save(onnx_model, model_path)


_model_ = net_ConstantOfShape()

_inputs_ = (torch.randn(1, 2, 4, 4), torch.randn(1, 8, 4))
