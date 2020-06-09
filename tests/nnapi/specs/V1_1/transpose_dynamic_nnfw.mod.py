import dynamic_tensor
model = Model()
model_input_shape = [1, 2, 2, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape)
test_node_input = dynamic_layer.getTestNodeInput()

perms = Parameter("perms", "TENSOR_INT32", "{4}", [0, 2, 1, 3])

model_output = Output("output", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model_transpose = model.Operation("TRANSPOSE", test_node_input, perms).To(model_output)

model_input_data = [1.0, 2.0, 3.0, 4.0]
model_output_data = [1.0, 3.0, 2.0, 4.0]

Example({
  dynamic_layer.getModelInput(): model_input_data,
  dynamic_layer.getShapeInput(): model_input_shape,
  model_output: model_output_data,
}, model=model_transpose)
