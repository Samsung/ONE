import torch
import onnx

INT8_MIN = -128
INT8_MAX = 127


# Generate QuantizeLinear/DequantizeLinear operator with Float32, Rank-3, layer-wise, INT16
class net_QuantizeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = (torch.randn(1) - 0.5) * 0.05
        self.zero_points = ((torch.randn(1) - 0.5) * 32).to(torch.int32)

    def forward(self, input):
        return torch.fake_quantize_per_tensor_affine(input, self.scales[0],
                                                     self.zero_points[0], INT8_MIN,
                                                     INT8_MAX)

    def onnx_opset_version(self):
        # TODO set version
        return 13

    def post_process(self, model_path):
        # check QuantizeLinear_F32_R4_i16_cw comments

        onnx_model = onnx.load(model_path)
        for node in onnx_model.graph.node:
            print(node)

        # find int8 constant
        node_int8_const = None
        for node in onnx_model.graph.node:
            if node.op_type == "Constant":
                for attribute in node.attribute:
                    if attribute.name == 'value' and attribute.t.data_type == onnx.TensorProto.INT8:
                        node_int8_const = node

        if node_int8_const == None:
            raise Exception("int8 node not found")

        i16_const_name = "/Constant_2_output_0"
        zero_points = ((torch.randn(1) - 0.5) * 32).to(torch.int16)
        ctensor = onnx.helper.make_tensor(name=i16_const_name,
                                          data_type=onnx.TensorProto.INT16,
                                          dims=[],
                                          vals=zero_points)
        Ci32 = onnx.helper.make_node("Constant", [], [i16_const_name],
                                     name=i16_const_name,
                                     value=ctensor)
        onnx_model.graph.node.insert(1, Ci32)

        # replace to use int16 constant
        for node in onnx_model.graph.node:
            if node.op_type == "QuantizeLinear":
                node.input[2] = i16_const_name

            if node.op_type == "DequantizeLinear":
                node.input[2] = i16_const_name

        onnx_model.graph.node.remove(node_int8_const)

        onnx.save(onnx_model, model_path)


_model_ = net_QuantizeLinear()

_inputs_ = torch.randn(1, 16, 16)
