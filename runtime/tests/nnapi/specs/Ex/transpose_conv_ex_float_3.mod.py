# model
model = Model()
i0 = Input("op_shape", "TENSOR_INT32", "{4}")
weights = Parameter("ker", "TENSOR_FLOAT32", "{2, 3, 3, 1}",
  [1.0, 3.0, 5.0, 7.0, 9.0, 11.0,
	 13.0, 15.0, 17.0, 2.0, 4.0, 6.0,
	 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}" )
pad = Int32Scalar("pad_valid", 2)
s_x = Int32Scalar("stride_x", 2)
s_y = Int32Scalar("stride_y", 2)
i2 = Output("op", "TENSOR_FLOAT32", "{1, 5, 5, 2}")
model = model.Operation("TRANSPOSE_CONV_EX", i0, weights, i1, pad, s_x, s_y).To(i2)

# Example 1. Input in operand 0,
input0 = {i0:  # output shape
          [1, 5, 5, 2],
          i1:  # input 0
          [1.0, 2.0, 3.0, 4.0]}

output0 = {i2:  # output 0
           [1.0, 2.0, 3.0, 4.0, 7.0,
	    10.0, 6.0, 8.0, 10.0, 12.0,
	    7.0, 8.0, 9.0, 10.0, 25.0,
	    28.0, 18.0, 20.0, 22.0, 24.0,
	    16.0, 20.0, 24.0, 28.0, 62.0,
	    72.0, 42.0, 48.0, 54.0, 60.0,
	    21.0, 24.0, 27.0, 30.0, 61.0,
	    68.0, 36.0, 40.0, 44.0, 48.0,
	    39.0, 42.0, 45.0, 48.0, 103.0,
	    110.0, 60.0, 64.0, 68.0, 72.0]}

# Instantiate an example
Example((input0, output0))
