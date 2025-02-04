model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 2, 4, 1}")
axis = Parameter("axis", "TENSOR_INT32", "{1}", [-3])
output = Output("output", "TENSOR_INT32", "{1, 4, 1}")

model = model.Operation("ARGMAX", i1, axis).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 7.0, 8.0,
           1.0, 9.0, 7.0, 3.0]}

output0 = {output: # output 0
          [0, 1, 0, 0]}

# Instantiate an example
Example((input0, output0))
