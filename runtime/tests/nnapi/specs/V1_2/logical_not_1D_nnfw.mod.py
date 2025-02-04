# model
model = Model()
i = Input("op1", "TENSOR_BOOL8", "{4}")

o = Output("op2", "TENSOR_BOOL8", "{4}")
model = model.Operation("LOGICAL_NOT", i).To(o)

# Example 1. Input
input0 = {i: # input
          [True, False, True, True]}

output0 = {o: # output
          [False, True, False, False]}

# Instantiate an example
Example((input0, output0))
