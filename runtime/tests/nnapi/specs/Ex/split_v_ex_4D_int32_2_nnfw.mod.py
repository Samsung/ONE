# model
model = Model()
i1 = Input("op1", "TENSOR_INT32", "{2,2,2,2}")
size_splits = Input("size_splits", "TENSOR_INT32", "{2}")
split_dim = Input("split_dim", "TENSOR_INT32", "{1}")
num_splits = Int32Scalar("num_splits", 2)

i2 = Output("op2", "TENSOR_INT32", "{2,1,2,2}")
i3 = Output("op3", "TENSOR_INT32", "{2,1,2,2}")

model = model.Operation("SPLIT_V_EX", i1, size_splits, split_dim, num_splits).To([i2, i3])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
          size_splits:
          [8, 8],
          split_dim:
          [1]}

output0 = {
    i2: # output 0
          [1, 2, 3, 4, 9, 10, 11, 12],
    i3: # output 1
            [5, 6, 7, 8, 13, 14, 15, 16]}

# Instantiate an example
Example((input0, output0))
