# model
model = Model()
i1 = Input("op1", "TENSOR_BOOL8", "{2, 1, 2, 2}")
i2 = Input("op2", "TENSOR_BOOL8", "{2, 1, 2, 2}")

i3 = Output("op3", "TENSOR_BOOL8", "{2, 1, 2, 2}")
model = model.Operation("LOGICAL_AND", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [False, False, True, True, False, True, False, True],
          i2: # input 1
          [False, True, False, True, False, False, True, True]}

output0 = {i3: # output 0
          [False, False, False, True, False, False, False, True]}

# Instantiate an example
Example((input0, output0))
