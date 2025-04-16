import torch
import onnx
import numpy as np

NUM_FEATURES = 5


# Generate InstanceNormalization operator with Float32, Rank-4
# Update network with scale/B constants with random value
class net_InstanceNorm2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.InstanceNorm2d(NUM_FEATURES)

    def forward(self, input):
        return self.op(input)

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # create two constants with random value
        shape = [NUM_FEATURES]
        new_s_name = '/op/Constant2_out'
        new_B_name = '/op/Constant3_out'
        new_s_v = onnx.helper.make_tensor(
            name=new_s_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=shape,
            vals=np.random.randn(NUM_FEATURES).flatten().astype(np.float32))
        new_B_v = onnx.helper.make_tensor(
            name=new_B_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=shape,
            vals=np.random.randn(NUM_FEATURES).flatten().astype(np.float32))

        new_s = onnx.helper.make_node("Constant", [], [new_s_name],
                                      name='/op/Constant2',
                                      value=new_s_v)
        new_B = onnx.helper.make_node("Constant", [], [new_B_name],
                                      name='/op/Constant3',
                                      value=new_B_v)

        # add new constants to graph
        onnx_model.graph.node.insert(0, new_s)
        onnx_model.graph.node.insert(1, new_B)

        # change InstanceNorm input scale and B to two new constants
        for node in onnx_model.graph.node:
            if node.op_type == "InstanceNormalization":
                node.input[1] = new_s_name
                node.input[2] = new_B_name

        onnx.save(onnx_model, model_path)


_model_ = net_InstanceNorm2d()

_inputs_ = torch.randn(1, NUM_FEATURES, 3, 3)
