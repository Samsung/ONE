batch = 2
rows = 3
cols = 4
depth = 5

input_table = [x for x in range(batch * rows * cols * depth)]
for i in range(batch):
  for j in range(rows):
    for k in range(cols):
      for l in range(depth):
        input_table[i * rows * cols * depth + j * cols * depth + k * depth + l] = i * rows * cols * depth + j * cols * depth + k * depth + l;

# Since the axises to be reduced are {rows, cols} and the value of the input always increases in here, the output's values are i * rows * cols * depth + (rows - 1) * cols * depth + (cols - 1) * depth + l.
output_table = [x for x in range(batch * depth)]
for i in range(batch):
  for l in range(depth):
    output_table[i * depth + l] = i * rows * cols * depth + (rows - 1) * cols * depth + (cols - 1) * depth + l;

model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{%d, %d, %d, %d}" % (batch, rows, cols, depth))
axis = Parameter("axis", "TENSOR_INT32", "{4}", [1, 2, -3, -2])
keepDims = False
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (batch, depth))

model = model.Operation("REDUCE_MAX", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          input_table}

output0 = {output: # output 0
          output_table}

# Instantiate an example
Example((input0, output0))
