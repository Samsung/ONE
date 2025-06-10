batch = 2
rows = 3
cols = 4
depth = 5

input_table = [x for x in range(batch * rows * cols * depth)]

output_table = [1 for x in range(batch * depth)]
for i in range(batch):
  for j in range(rows):
    for k in range(cols):
      for l in range(depth):
        # The value of output_table is the rowwise sum and colwise sum of input_table.
        output_table[i * depth + l] *= input_table[i * rows * cols * depth + j * cols * depth + k * depth + l];

model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{%d, %d, %d, %d}" % (batch, rows, cols, depth))
# Axis value should be in the range [-(rank), rank). And '-n' is the same axis with 'rank - n'. So this test's axis value are the same [1, 2].
axis = Parameter("axis", "TENSOR_INT32", "{4}", [1, 2, -3, -2])
keepDims = False
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (batch, depth))

model = model.Operation("REDUCE_PROD", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          input_table}

output0 = {output: # output 0
          output_table}

# Instantiate an example
Example((input0, output0))
