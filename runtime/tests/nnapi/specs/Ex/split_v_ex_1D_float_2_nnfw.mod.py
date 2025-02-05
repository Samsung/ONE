# model
input0 = Input("input0", "TENSOR_FLOAT32", "{12}")
size_splits = Input("size_splits", "TENSOR_INT32", "{3}")
split_dim = Input("split_dim", "TENSOR_INT32", "{1}") 
num_splits = Int32Scalar("num_splits",3);

output0 = Output("output0", "TENSOR_FLOAT32", "{3}")
output1 = Output("output1", "TENSOR_FLOAT32", "{5}")
output2 = Output("output2", "TENSOR_FLOAT32", "{4}")

model = Model().Operation("SPLIT_V_EX", input0, size_splits, split_dim, num_splits).To((output0, output1, output2))

# Example 1.
input_dict = {
    input0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    size_splits: [3, 5, 4],
    split_dim: [0]
}
output_dict = {
    output0: [1.0, 2.0, 3.0],
    output1: [4.0, 5.0, 6.0, 7.0, 8.0],
    output2: [9.0, 10.0, 11.0, 12.0]
}

Example((input_dict, output_dict))
