model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 2, 2, 8}")
block = Int32Scalar("block_size", 2)
output = Output("output", "TENSOR_FLOAT32", "{1, 4, 4, 2}")

model = model.Operation("DEPTH_TO_SPACE", i1, block).To(output)

# Example 1. Input in operand 0,

input0 = {i1: # input 0
           [10,   20,  11,  21, 14,   24,  15,  25,
            12,   22,  13,  23, 16,   26,  17,  27,
            18,   28,  19,  29, 112, 212, 113, 213,
            110, 210, 111, 211, 114, 214, 115, 215]}

output0 = {output: # output 0
          [10,   20,  11,  21,  12,  22, 13,   23,
           14,   24,  15,  25,  16,  26, 17,   27,
           18,   28,  19,  29, 110, 210, 111, 211,
          112,  212, 113, 213, 114, 214, 115, 215]}
# Instantiate an example
Example((input0, output0))
