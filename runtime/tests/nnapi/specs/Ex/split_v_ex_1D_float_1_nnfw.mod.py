# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{8}")
size_splits = Input("size_splits", "TENSOR_INT32", "{8}")
split_dim = Input("split_dim", "TENSOR_INT32", "{1}")
num_splits = Int32Scalar("num_splits", 8)

i2 = Output("op2", "TENSOR_FLOAT32", "{1}")
i3 = Output("op3", "TENSOR_FLOAT32", "{1}")
i4 = Output("op4", "TENSOR_FLOAT32", "{1}")
i5 = Output("op5", "TENSOR_FLOAT32", "{1}")
i6 = Output("op6", "TENSOR_FLOAT32", "{1}")
i7 = Output("op7", "TENSOR_FLOAT32", "{1}")
i8 = Output("op8", "TENSOR_FLOAT32", "{1}")
i9 = Output("op9", "TENSOR_FLOAT32", "{1}")

model = model.Operation("SPLIT_V_EX", i1, size_splits, split_dim, num_splits).To([i2, i3, i4, i5, i6, i7, i8, i9])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          size_splits:
          [1, 1, 1, 1, 1, 1, 1, 1],
          split_dim:
          [0]
          }

output0 = {
    i2: # output 0
          [1.0],
    i3: # output 1
          [2.0],
    i4: # output 2
          [3.0],
    i5: # output 3
          [4.0],
    i6: # output 4
          [5.0],
    i7: # output 5
          [6.0],
    i8: # output 6
          [7.0],
    i9: # output 7
          [8.0]}

# Instantiate an example
Example((input0, output0))
