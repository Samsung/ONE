model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
block = Int32Scalar("block_size", 2)
output = Output("output", "TENSOR_FLOAT32", "{1, 1, 1, 8}")

model = model.Operation("SPACE_TO_DEPTH", i1, block).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]}

output0 = {output: # output 0
           [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]}

# Instantiate an example
Example((input0, output0))
