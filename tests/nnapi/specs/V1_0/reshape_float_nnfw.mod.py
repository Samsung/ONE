# model
model = Model()
input_ = Input("op1", "TENSOR_FLOAT32", "{1, 1, 3, 3}")
param_ = Parameter("op2", "TENSOR_INT32", "{2}", [1, -1])
output_ = Output("op3", "TENSOR_FLOAT32", "{1, 9}")
model = model.Operation("RESHAPE", input_, param_).To(output_)

# Example 1. Input in operand 0,
input0 = {input_: # input 0
          [1, 2, 3,
           4, 5, 6,
           7, 8, 9]}

output0 = {output_: # output 0
           [1, 2, 3, 4, 5, 6, 7, 8, 9]}

# Instantiate an example
Example((input0, output0))
