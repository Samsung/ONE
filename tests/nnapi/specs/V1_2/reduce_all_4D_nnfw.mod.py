# model
model = Model()
i1 = Input("input", "TENSOR_BOOL8", "{1, 3, 4, 1}")
axis = Int32Scalar("axis", 1)
keepDims = True
out1 = Output("output", "TENSOR_BOOL8", "{1, 1, 4, 1}")
model = model.Operation("REDUCE_ALL", i1, axis, keepDims).To(out1)

# Example 1. Input in operand 0, 1
input0 = {i1: # input 0
          [True, True, False, False,
           True, False, False ,True,
           True, False, False, True]}

output0 = {out1: # output 0
           [True, False, False, False]}

# Instantiate an example
Example((input0, output0))
