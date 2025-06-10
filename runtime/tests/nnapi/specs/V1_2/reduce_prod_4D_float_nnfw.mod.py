# model
model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 3, 4, 1}")
axis = Int32Scalar("axis", 1)
keepDims = False
out1 = Output("output", "TENSOR_FLOAT32", "{1, 4, 1}")
model = model.Operation("REDUCE_PROD", i1, axis, keepDims).To(out1)

# Example 1. Input in operand 0, 1
input0 = {i1: # input 0
          [6.4, 7.3, 19.3, -2.3,
           8.3, 2.0, 11.8, -3.4,
           22.8, 3.0, -28.7, 4.9]}

output0 = {out1: # output 0
           [1211.136,	43.8,	-6536.138,	38.318]}

# Instantiate an example
Example((input0, output0))
