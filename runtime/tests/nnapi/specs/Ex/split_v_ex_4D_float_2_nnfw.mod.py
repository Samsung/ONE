# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2,2,2,2}")
size_splits = Input("size_splits", "TENSOR_INT32", "{2}")
split_dim = Input("split_dim", "TENSOR_INT32", "{1}")
num_splits = Int32Scalar("num_splits", 2)

i2 = Output("op2", "TENSOR_FLOAT32", "{2,2,2,1}")
i3 = Output("op3", "TENSOR_FLOAT32", "{2,2,2,1}")
model = model.Operation("SPLIT_V_EX", i1, size_splits, split_dim, num_splits).To([i2, i3])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
          size_splits:
          [8, 8],
          split_dim:
          [3]}

output0 = {
    i2: # output 0
          [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
    i3: # output 1
          [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]}

# Instantiate an example
Example((input0, output0))
