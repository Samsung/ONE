# model
model = Model()
i1 = Input("input", "TENSOR_INT32", "{3, 4}")
axis = Int32Scalar("axis", 1)
keepDims = False
out1 = Output("output", "TENSOR_INT32", "{3}")
model = model.Operation("REDUCE_MAX", i1, axis, keepDims).To(out1)

# Example 1. Input in operand 0, 1
input0 = {i1: # input 0
          [3, 11, 3, 5,
           28, 0, -1, -13,
           -4, -22, -2, -49]}

output0 = {out1: # output 0
           [11, 28, -2]}

# Instantiate an example
Example((input0, output0))
