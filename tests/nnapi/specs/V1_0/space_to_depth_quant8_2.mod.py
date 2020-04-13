model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{1, 4, 4, 1}, 0.5f, 0")
block = Int32Scalar("radius", 2)
output = Output("output", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 4}, 0.5f, 0")

model = model.Operation("SPACE_TO_DEPTH", i1, block).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255]}

output0 = {output: # output 0
           [0, 1, 4, 5, 2, 3, 6, 7, 248, 249, 252, 253, 250, 251, 254, 255]}

# Instantiate an example
Example((input0, output0))
