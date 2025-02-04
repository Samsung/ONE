# model
import dynamic_tensor
model = Model()

start_shape = [1]
dynamic_start = dynamic_tensor.DynamicInputGenerator(model, start_shape)
test_start_input = dynamic_start.getTestNodeInput()

limit_shape = [1]
dynamic_limit = dynamic_tensor.DynamicInputGenerator(model, limit_shape)
test_limit_input = dynamic_limit.getTestNodeInput()

delta_shape = [1]
dynamic_delta = dynamic_tensor.DynamicInputGenerator(model, delta_shape)
test_delta_input = dynamic_delta.getTestNodeInput()


out = Output("output", "TENSOR_FLOAT32", "{8}") 
model = model.Operation("RANGE_EX", test_start_input, test_limit_input, test_delta_input).To(out)


test_start_data = [1]
test_limit_data = [5]
test_delta_data = [0.5]
output_data = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


# Instantiate an example
Example({
  dynamic_start.getModelInput() : test_start_data,
  dynamic_start.getShapeInput() : start_shape,
  dynamic_limit.getModelInput() : test_limit_data,
  dynamic_limit.getShapeInput() : limit_shape,
  dynamic_delta.getModelInput() : test_delta_data,
  dynamic_delta.getShapeInput() : delta_shape,
  out                           : output_data
})
