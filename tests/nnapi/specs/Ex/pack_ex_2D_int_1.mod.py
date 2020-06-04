# Sample Stack or Pack model
model = Model()
i1 = Input("input1", "TENSOR_INT32", "{6, 4}")
i2 = Input("input2", "TENSOR_INT32", "{6, 4}")
i3 = Input("input3", "TENSOR_INT32", "{6, 4}")
num = Int32Scalar("num_tensors", 3)
axis = Int32Scalar("axis", 0)
out = Output("output", "TENSOR_INT32", "{3, 6, 4}")
model = model.Operation("PACK_EX", i1, i2, i3, num, axis).To(out)

input0 = {i1: # input 0
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
         i2: # input 1
         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
         i3: # input 2
         [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]}

output0 = {out: # output 0
           [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
             54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]}

# Instantiate an example
Example((input0, output0))
