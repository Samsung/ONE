# model
model = Model()
i1 = Input("op1", "TENSOR_BOOL8", "{1, 2, 2, 1}")
i2 = Input("op2", "TENSOR_BOOL8", "{1, 1, 2, 1}")

i3 = Output("op3", "TENSOR_BOOL8", "{1, 2, 2, 1}")
model = model.Operation("LOGICAL_AND", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [False, False, True, True],
          i2: # input 1
          [False, True]}

output0 = {i3: # output 0
          [False, False, False, True]}

# Instantiate an example
Example((input0, output0))
