model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{1, 4, 2, 1}, 1.0, 9")
block = Parameter("block_size", "TENSOR_INT32", "{2}", [3, 2])
paddings = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [1, 1, 2, 4])
output = Output("output", "TENSOR_QUANT8_ASYMM", "{6, 2, 4, 1}, 1.0, 9")

model = model.Operation("SPACE_TO_BATCH_ND", i1, block, paddings).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8]}

output0 = {output: # output 0
           [9, 9, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9,
            9, 1, 9, 9, 9, 7, 9, 9, 9, 2, 9, 9, 9, 8, 9, 9,
            9, 3, 9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9]}

# Instantiate an example
Example((input0, output0))
