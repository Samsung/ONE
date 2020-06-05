model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
block = Int32Scalar("block_size", 2)
output = Output("output", "TENSOR_FLOAT32", "{1, 4, 4, 1}")

model = model.Operation("DEPTH_TO_SPACE", i1, block).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
           [1., 2., 5., 6., 3., 4., 7., 8., 9., 10., 13., 14., 11., 12., 15., 16.]}

output0 = {output: # output 0
          [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]}

# Instantiate an example
Example((input0, output0))
