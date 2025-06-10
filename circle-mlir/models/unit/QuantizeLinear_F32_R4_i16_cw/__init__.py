import torch
import onnx

INT8_MIN = -128
INT8_MAX = 127

CHN_NUM = 4
CHN_AXIS = 1


# Generate QuantizeLinear/DequantizeLinear operator with Float32, Rank-4, channel-wise, INT16
class net_QuantizeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = (torch.randn(CHN_NUM) - 0.5) * 0.05
        self.zero_points = ((torch.randn(CHN_NUM) - 0.5) * 32).to(torch.int32)

    def forward(self, input):
        # as torch.onnx cannot produce INT16,
        # save this as INT8 and then convert it to INT16 in post_process()
        return torch.fake_quantize_per_channel_affine(input, self.scales,
                                                      self.zero_points, CHN_AXIS,
                                                      INT8_MIN, INT8_MAX)

    def onnx_opset_version(self):
        # TODO set version
        return 13

    def post_process(self, model_path):
        # what this does
        # 1/ create Constant with int16 to replace int8
        # 2/ update QuantizeLinear with
        #    - change 3rd, y_zero_point, input to int16 constant
        #    - change output dtype to int16 -> no need
        # 3/ update DequantizeLinear with
        #    - change 3rd, x_zero_point, input to int16 constant
        #    - change input dtype to int16 -> no need
        onnx_model = onnx.load(model_path)

        # find int8 constant
        node_int8_const = None
        for node in onnx_model.graph.node:
            if node.op_type == "Constant":
                for attribute in node.attribute:
                    if attribute.name == 'value' and attribute.t.data_type == onnx.TensorProto.INT8:
                        node_int8_const = node

        if node_int8_const == None:
            raise Exception("int8 node not found")

        # node_int8_const name is "/Constant_1_output_0", just use this convention
        i16_const_name = "/Constant_2_output_0"
        # we can read node_int8_const attributes but just use what we know
        zero_points = ((torch.randn(CHN_NUM) - 0.5) * 32).to(torch.int16)
        ctensor = onnx.helper.make_tensor(name=i16_const_name,
                                          data_type=onnx.TensorProto.INT16,
                                          dims=[CHN_NUM],
                                          vals=zero_points)
        Ci32 = onnx.helper.make_node("Constant", [], [i16_const_name],
                                     name=i16_const_name,
                                     value=ctensor)
        onnx_model.graph.node.insert(2, Ci32)

        # replace to use int16 constant
        for node in onnx_model.graph.node:
            if node.op_type == "QuantizeLinear":
                node.input[2] = i16_const_name

            if node.op_type == "DequantizeLinear":
                node.input[2] = i16_const_name

        onnx_model.graph.node.remove(node_int8_const)

        # QuantizeLinear/DequantizeLinear output node type follows 2'nd constant
        # so we don't need to change output data type

        onnx.save(onnx_model, model_path)


_model_ = net_QuantizeLinear()

_inputs_ = torch.randn(1, CHN_NUM, 16, 9)
