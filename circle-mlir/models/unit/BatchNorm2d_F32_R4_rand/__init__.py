import torch
import onnx
import numpy as np

NUM_FEATURES = 5


# Generate BatchNorm2d operator with Float32, Rank-4 with random value constants
# NOTE affine=True(default) will generate mean/var to input, not constant
class net_BatchNorm2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.BatchNorm2d(NUM_FEATURES, affine=False)

    def forward(self, input):
        return self.op(input)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 14

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # create four constants with random value
        shape = [NUM_FEATURES]
        new_s_name = '/op/Constant_output_0_n'
        new_B_name = '/op/Constant_1_output_0_n'
        new_m_name = '/op/running_mean_n'
        new_v_name = '/op/running_var_n'

        vals = np.clip(
            0.9 - np.random.randn(NUM_FEATURES).flatten().astype(np.float32) / 20.0, 0.0,
            1.0)
        new_s_v = onnx.helper.make_tensor(name=new_s_name,
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=shape,
                                          vals=vals)
        vals = np.clip(
            0.1 + np.random.randn(NUM_FEATURES).flatten().astype(np.float32) / 20.0, 0.0,
            1.0)
        new_B_v = onnx.helper.make_tensor(name=new_B_name,
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=shape,
                                          vals=vals)
        vals = np.clip(
            0.1 + np.random.randn(NUM_FEATURES).flatten().astype(np.float32) / 20.0, 0.0,
            1.0)
        new_m_v = onnx.helper.make_tensor(name=new_m_name,
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=shape,
                                          vals=vals)
        vals = np.clip(
            0.9 - np.random.randn(NUM_FEATURES).flatten().astype(np.float32) / 20.0, 0.0,
            1.0)
        new_v_v = onnx.helper.make_tensor(name=new_v_name,
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=shape,
                                          vals=vals)

        new_s = onnx.helper.make_node("Constant", [], [new_s_name],
                                      name='/op/Constant_output_0_n',
                                      value=new_s_v)
        new_B = onnx.helper.make_node("Constant", [], [new_B_name],
                                      name='/op/Constant_1_output_0_n',
                                      value=new_B_v)
        new_m = onnx.helper.make_node("Constant", [], [new_m_name],
                                      name='/op/running_mean_n',
                                      value=new_m_v)
        new_v = onnx.helper.make_node("Constant", [], [new_v_name],
                                      name='/op/running_var_n',
                                      value=new_v_v)

        # add new constants to graph
        onnx_model.graph.node.insert(0, new_s)
        onnx_model.graph.node.insert(1, new_B)
        onnx_model.graph.node.insert(2, new_m)
        onnx_model.graph.node.insert(3, new_v)

        # change BatchNormalization inputs to four new constants
        for node in onnx_model.graph.node:
            if node.op_type == "BatchNormalization":
                node.input[1] = new_s_name
                node.input[2] = new_B_name
                node.input[3] = new_m_name
                node.input[4] = new_v_name

        onnx.save(onnx_model, model_path)


_model_ = net_BatchNorm2d()

_inputs_ = torch.randn(1, NUM_FEATURES, 4, 4)
