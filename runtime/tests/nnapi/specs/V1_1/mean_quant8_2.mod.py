model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{4, 3, 2}, 0.8, 5")
axis = Parameter("axis", "TENSOR_INT32", "{2}", [0, 2])
keepDims = Int32Scalar("keepDims", 1)
output = Output("output", "TENSOR_QUANT8_ASYMM", "{1, 3, 1}, 0.8, 5")

model = model.Operation("MEAN", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1,  2,  3,  4,  5,  6,  7,  8,
           9,  10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24]}

output0 = {output: # output 0
          [10, 12, 14]}

# Instantiate an example
Example((input0, output0))
