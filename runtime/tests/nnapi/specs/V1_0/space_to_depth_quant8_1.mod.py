model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 2}, 0.5f, 0")
block = Int32Scalar("radius", 2)
output = Output("output", "TENSOR_QUANT8_ASYMM", "{1, 1, 1, 8}, 0.5f, 0")

model = model.Operation("SPACE_TO_DEPTH", i1, block).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 252, 253, 254, 255]}

output0 = {output: # output 0
           [1, 2, 3, 4, 252, 253, 254, 255]}

# Instantiate an example
Example((input0, output0))
