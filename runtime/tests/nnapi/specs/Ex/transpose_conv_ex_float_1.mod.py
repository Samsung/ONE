# model
model = Model()
i0 = Input("op_shape", "TENSOR_INT32", "{4}")
weights = Parameter("ker", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
i1 = Input("in", "TENSOR_FLOAT32", "{1, 4, 4, 1}" )
pad = Int32Scalar("pad_same", 1)
s_x = Int32Scalar("stride_x", 1)
s_y = Int32Scalar("stride_y", 1)
i2 = Output("op", "TENSOR_FLOAT32", "{1, 4, 4, 1}")
model = model.Operation("TRANSPOSE_CONV_EX", i0, weights, i1, pad, s_x, s_y).To(i2)

# Example 1. Input in operand 0,
input0 = {i0:  # output shape
          [1, 4, 4, 1],
          i1:  # input 0
          [1.0, 2.0, 3.0, 4.0,
	   5.0, 6.0, 7.0, 8.0,
	   9.0, 10.0, 11.0, 12.0,
	   13.0, 14.0, 15.0, 16.0]}

output0 = {i2:  # output 0
           [29.0, 62.0, 83.0, 75.0,
 	    99.0, 192.0, 237.0, 198.0,
	    207.0, 372.0, 417.0, 330.0,
	    263.0, 446.0, 485.0, 365.0]}

# Instantiate an example
Example((input0, output0))
