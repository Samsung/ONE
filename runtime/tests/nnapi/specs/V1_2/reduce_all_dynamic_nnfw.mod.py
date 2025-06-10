import dynamic_tensor

model = Model()

model_input_shape = [1, 3, 4, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_BOOL8")
test_node_input = dynamic_layer.getTestNodeInput()

# write REDUCE_ALL test. input is `test_input`

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_BOOL8", "{1, 1, 4, 1}")

axis = Int32Scalar("axis", 1)
keepDims = True

model.Operation("REDUCE_ALL", test_node_input, axis, keepDims).To(model_output)

model_input_data = [True, True, False, False,
           True, False, False ,True,
           True, False, False, True]

model_output_data = [True, False, False, False]

Example({
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
