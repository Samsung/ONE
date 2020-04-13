model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{3,4}")
perms = Parameter("perms", "TENSOR_INT32", "{2}", [1, 0])
output = Output("output", "TENSOR_FLOAT32", "{4,3}")

model = model.Operation("TRANSPOSE", i1, perms).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

output0 = {output: # output 0
          [0, 4, 8, 1, 5, 9, 2,  6, 10,  3,  7, 11]}

# Instantiate an example
Example((input0, output0))
