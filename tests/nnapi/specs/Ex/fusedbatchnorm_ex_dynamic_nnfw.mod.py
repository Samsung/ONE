import dynamic_tensor

# model
model = Model()

model_input_shape = [1, 1, 4, 1]
dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()
scale = Input("scale", "TENSOR_FLOAT32", "{1}")
offset = Input("offset", "TENSOR_FLOAT32", "{1}")
mean = Input("mean", "TENSOR_FLOAT32", "{1}")
variance = Input("variance", "TENSOR_FLOAT32", "{1}")
o1 = Output("mean", "TENSOR_FLOAT32", "{1,1,4,1}")

is_training = BoolScalar("is_training", True)
data_format = Int32Vector("data_format", [4]) # TODO: support asymm8 as below
#data_format = Parameter("data_format", "TENSOR_QUANT8_ASYMM", "{4}, 1.0, 0",
#                        [78, 72, 87, 67]), # NHWC: nnapi always assumes channel-last layout
epsilon = Float32Scalar("epsilon", 0)

model = model.Operation("FUSED_BATCH_NORM_V3_EX",
                          test_node_input, scale, offset, mean, variance,              # inputs
                          is_training, data_format, epsilon).To(o1)  # param


input0 = {dynamic_layer.getModelInput(): [0., 1., -1., 0.],
          dynamic_layer.getShapeInput() : model_input_shape,
          scale: [1.],
          offset: [0.],
          mean: [0.],
          variance: [1.],
}

output0 = {o1 :
           [0, 1.4142135381698608 , -1.4142135381698608, 0]}

# Instantiate an example
Example((input0, output0))
