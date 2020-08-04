# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{4,1,1,8}")
size_splits = Input("size_splits", "TENSOR_INT32", "{3}")
split_dim = Input("split_dim", "TENSOR_INT32", "{1}")
num_splits = Int32Scalar("num_splits", 3)

i2 = Output("op2", "TENSOR_FLOAT32", "{4,1,1,2}")
i3 = Output("op3", "TENSOR_FLOAT32", "{4,1,1,4}")
i4 = Output("op4", "TENSOR_FLOAT32", "{4,1,1,2}")

model = model.Operation("SPLIT_V_EX", i1, size_splits, split_dim, num_splits).To([i2, i3, i4])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
          size_splits:
          [2,4,2],
          split_dim:
          [3]
          }

output0 = {
    i2: # output 0
          [1.0, 2.0, 9.0, 10.0, 17.0, 18.0, 25.0, 26.0],
    i3: # output 1
          [3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0, 27.0, 28.0, 29.0, 30.0],
    i4: [7.0, 8.0, 15.0, 16.0, 23.0, 24.0, 31.0, 32.0]}

# Instantiate an example
Example((input0, output0))
