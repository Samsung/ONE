import dynamic_tensor

# model
model = Model()

model_input_shape = [1, 1, 4, 4]
dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")
test_node_input = dynamic_layer.getTestNodeInput()

i2 = Input("op2", "TENSOR_INT32", "{1}")
i3 = Input("op3", "TENSOR_INT32", "{1}")
i4 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 4, 4}")
model = model.Operation("MATRIX_BAND_PART_EX", test_node_input, i2, i3).To(i4)

input0 = {dynamic_layer.getModelInput():
          [0, 1, 2, 3, -1, 0, 1, 2, -2, -1, 0, 1, -3, -2, -1, 0],
          dynamic_layer.getShapeInput() : model_input_shape,
          i2: # input 1
          [1],
          i3: # input 2
          [-1]}

output0 = {i4: # output 0
           [0, 1, 2, 3, -1, 0, 1, 2, 0, -1,  0, 1, 0,  0, -1, 0]}

# Instantiate an example
Example((input0, output0))
