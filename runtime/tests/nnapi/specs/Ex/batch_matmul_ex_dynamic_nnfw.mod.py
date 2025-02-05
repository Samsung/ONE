import dynamic_tensor

model = Model()

input1_shape = [2, 2, 3]
dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input1_shape, "TENSOR_FLOAT32")

input1 = dynamic_layer.getTestNodeInput()
input2 = Input("op2", "TENSOR_FLOAT32", "{3, 4}")
adj_x = False
adj_y = False
model_output = Output("output", "TENSOR_FLOAT32", "{2, 2, 4}")

model = model.Operation("BATCH_MATMUL_EX", input1, input2, adj_x, adj_y).To(model_output)

input1_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
input2_data = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
model_output_data = [74, 80, 86, 92, 173, 188, 203, 218, 272, 296,
    320, 344, 371, 404, 437, 470]

input_list = {
    dynamic_layer.getModelInput(): input1_data,
    dynamic_layer.getShapeInput() : input1_shape,

    input2 : input2_data,
    }

output_list= {
    model_output: model_output_data
    }

Example((input_list, output_list))
