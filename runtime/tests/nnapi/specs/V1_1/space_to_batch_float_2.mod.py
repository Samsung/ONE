model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 5, 2, 1}")
block = Parameter("block_size", "TENSOR_INT32", "{2}", [3, 2])
paddings = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [1, 0, 2, 0])
output = Output("output", "TENSOR_FLOAT32", "{6, 2, 2, 1}")

model = model.Operation("SPACE_TO_BATCH_ND", i1, block, paddings).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

output0 = {output: # output 0
           [0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
            0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10]}

# Instantiate an example
Example((input0, output0))
