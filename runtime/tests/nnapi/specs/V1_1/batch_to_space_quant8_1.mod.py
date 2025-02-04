model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{4, 2, 2, 1}, 1.0, 0")
block = Parameter("block_size", "TENSOR_INT32", "{2}", [2, 2])
output = Output("output", "TENSOR_QUANT8_ASYMM", "{1, 4, 4, 1}, 1.0, 0")

model = model.Operation("BATCH_TO_SPACE_ND", i1, block).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}

output0 = {output: # output 0
           [1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16]}

# Instantiate an example
Example((input0, output0))
