# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{1, 1, 4, 4}")
i2 = Input("op2", "TENSOR_INT32", "{}")
i3 = Input("op3", "TENSOR_INT32", "{}")
i4 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 4, 4}")
model = model.Operation("MATRIX_BAND_PART_EX", i1, i2, i3).To(i4)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3, -1, 0, 1, 2, -2, -1, 0, 1, -3, -2, -1, 0],
          i2: # input 1
          [1],
          i3: # input 2
          [-1]}


output0 = {i4: # output 0
           [0, 1, 2, 3, -1, 0, 1, 2, 0, -1,  0, 1, 0,  0, -1, 0]}

# Instantiate an example
Example((input0, output0))


# Example 2. Input in operand 0,
input1 = {i1: # input 0
          [0, 1, 2, 3, -1, 0, 1, 2, -2, -1, 0, 1, -3, -2, -1, 0],
          i2: # input 1
          [2],
          i3: # input 2
          [1]}


output1 = {i4: # output 0
           [0, 1, 0, 0, -1, 0, 1, 0, -2, -1,  0, 1, 0, -2, -1, 0]}

Example((input1, output1))
