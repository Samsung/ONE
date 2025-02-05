# model
model = Model()
i = Input("op1", "TENSOR_BOOL8", "{1, 2, 2, 1}") # a vector of input

o = Output("op2", "TENSOR_BOOL8", "{1, 2, 2, 1}") # a vector of output
model = model.Operation("LOGICAL_NOT", i).To(o)

# Example 1. Input
input0 = {i: # input
          [False, True, True, True]}

output0 = {o: # output
          [True, False, False, False]}

# Instantiate an example
Example((input0, output0))
