model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{3,4,2}")
perms = Parameter("perms", "TENSOR_INT32", "{3}", [1, 0, 2])
output = Output("output", "TENSOR_FLOAT32", "{4,3,2}")

model = model.Operation("TRANSPOSE", i1, perms).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

output0 = {output: # output 0
          [ 0, 1, 8, 9, 16, 17, 2, 3, 10, 11, 18, 19, 4, 5, 12, 13, 20, 21, 6, 7, 14, 15, 22, 23]}

# Instantiate an example
Example((input0, output0))
