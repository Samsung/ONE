model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{1, 4, 4, 1}, 1.0, 0")
block = Parameter("block_size", "TENSOR_INT32", "{2}", [2, 2])
paddings = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [0, 0, 0, 0])
output = Output("output", "TENSOR_QUANT8_ASYMM", "{4, 2, 2, 1}, 1.0, 0")

model = model.Operation("SPACE_TO_BATCH_ND", i1, block, paddings).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}

output0 = {output: # output 0
           [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]}

# Instantiate an example
Example((input0, output0))
