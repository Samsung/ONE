# model
model = Model()
i0 = Input("op_shape", "TENSOR_INT32", "{4}")
weights = Parameter("ker", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}" )
pad = Int32Scalar("pad_valid", 2)
s_x = Int32Scalar("stride_x", 2)
s_y = Int32Scalar("stride_y", 2)
i2 = Output("op", "TENSOR_FLOAT32", "{1, 5, 5, 1}")
model = model.Operation("TRANSPOSE_CONV_EX", i0, weights, i1, pad, s_x, s_y).To(i2)

# Example 1. Input in operand 0,
input0 = {i0:  # output shape
          [1, 5, 5, 1],
          i1:  # input 0
          [1.0, 2.0, 3.0, 4.0]}

output0 = {i2:  # output 0
           [1.0, 2.0, 5.0, 4.0, 6.0,
            4.0, 5.0, 14.0, 10.0, 12.0,
	    10.0, 14.0, 36.0, 24.0, 30.0,
	    12.0, 15.0, 34.0, 20.0, 24.0,
	    21.0, 24.0, 55.0, 32.0, 36.0]}

# Instantiate an example
Example((input0, output0))
