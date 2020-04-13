model = Model()
i1 = Input("input", "TENSOR_FLOAT16", "{1, 2, 2, 1}")
axis = Parameter("axis", "TENSOR_INT32", "{1}", [2])
keepDims = Int32Scalar("keepDims", 0)
output = Output("output", "TENSOR_FLOAT16", "{1, 2, 1}")

model = model.Operation("MEAN", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0,
           3.0, 4.0]}

output0 = {output: # output 0
          [1.5,
           3.5]}

# Instantiate an example
Example((input0, output0))
