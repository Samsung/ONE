import dynamic_tensor

model = Model()
model_perm_shape = [4]
dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_perm_shape, tensor_type='TENSOR_INT32')
test_node_perm = dynamic_layer.getTestNodeInput()

inputs = Input("inputs", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model_output = Output("output", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model_transpose = model.Operation("TRANSPOSE", inputs, test_node_perm).To(model_output)

model_input_data = [1.0, 2.0, 3.0, 4.0]
model_output_data = [1.0, 3.0, 2.0, 4.0]
perms_data = [0, 2, 1, 3]

Example({
  inputs:model_input_data,
  dynamic_layer.getModelInput(): perms_data,
  dynamic_layer.getShapeInput(): model_perm_shape,
  model_output: model_output_data,
}, model=model_transpose)
