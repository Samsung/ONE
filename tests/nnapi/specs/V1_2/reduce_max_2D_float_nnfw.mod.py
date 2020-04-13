# model
model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{3, 4}")
axis = Int32Scalar("axis", 1)
keepDims = False
out1 = Output("output", "TENSOR_FLOAT32", "{3}")
model = model.Operation("REDUCE_MAX", i1, axis, keepDims).To(out1)

# Example 1. Input in operand 0, 1
input0 = {i1: # input 0
          [3.2, 11.47, 3.8, 5.76,
           28.2, 0.999, -1.3, -13.5,
           -3.4, -22.1, -2.2, -49.7]}

output0 = {out1: # output 0
           [11.47, 28.2, -2.2]}

# Instantiate an example
Example((input0, output0))
