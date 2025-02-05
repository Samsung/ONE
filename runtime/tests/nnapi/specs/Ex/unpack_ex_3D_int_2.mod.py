# Sample UnPack model, axis = 1
model = Model()
input = Input("input", "TENSOR_INT32", "{6, 3, 4}")
axis = Int32Scalar("axis", 1)
num_splits = Int32Scalar("num_splits", 3)
out1 = Output("output1", "TENSOR_INT32", "{6, 4}")
out2 = Output("output2", "TENSOR_INT32", "{6, 4}")
out3 = Output("output3", "TENSOR_INT32", "{6, 4}")
model = model.Operation("UNPACK_EX", input, num_splits, axis).To([out1, out2, out3])

input0 = {input: # input 0
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
	   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
	   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
	   54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]}

output0 = {out1: # output 0
          [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51, 60, 61, 62, 63],
	  out2: # output 1
	  [4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55, 64, 65, 66, 67],
	  out3: # output 2
	  [8, 9,10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59, 68, 69, 70, 71]}

# Instantiate an example
Example((input0, output0))
