model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{4, 1, 1, 2}")
squeezeDims = Parameter("squeezeDims", "TENSOR_INT32", "{2}", [1, 2])
output = Output("output", "TENSOR_FLOAT32", "{4, 2}")

model = model.Operation("SQUEEZE", i1, squeezeDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]}

output0 = {output: # output 0
           [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]}

# Instantiate an example
Example((input0, output0))
