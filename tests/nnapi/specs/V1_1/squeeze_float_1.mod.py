model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 24, 1}")
squeezeDims = Parameter("squeezeDims", "TENSOR_INT32", "{1}", [2])
output = Output("output", "TENSOR_FLOAT32", "{1, 24}")

model = model.Operation("SQUEEZE", i1, squeezeDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]}

output0 = {output: # output 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]}

# Instantiate an example
Example((input0, output0))
