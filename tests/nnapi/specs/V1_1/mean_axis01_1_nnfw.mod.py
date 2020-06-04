model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 4, 3, 2}")
axis = Parameter("axis", "TENSOR_INT32", "{2}", [1, 2])
keepDims = Int32Scalar("keepDims", 1)
output = Output("output", "TENSOR_FLOAT32", "{1, 1, 1, 2}")

model = model.Operation("MEAN", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0,
           1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0]}

output0 = {output: # output 0
          [1.0, 1.0]}

# Instantiate an example
Example((input0, output0))
