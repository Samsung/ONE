# model
model = Model()
i1 = Input("input", "TENSOR_BOOL8", "{3, 4}")
axis = Int32Scalar("axis", 1)
keepDims = False
out1 = Output("output", "TENSOR_BOOL8", "{3}")
model = model.Operation("REDUCE_ANY", i1, axis, keepDims).To(out1)

# Example 1. Input in operand 0, 1
input0 = {i1: # input 0
          [False, False, False, False,
           False, True, False, False,
           True, False, True, False]}

output0 = {out1: # output 0
           [False, True, True]}

# Instantiate an example
Example((input0, output0))
