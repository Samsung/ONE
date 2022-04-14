# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 2}")

i3 = Output("op3", "TENSOR_BOOL8", "{2, 2}")
model = model.Operation("EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [7.45, 3.21, 2.0, 7.67],
          i2: # input 1
          [0.0, 7.67]}

output0 = {i3: # output 0
           [0, 0, 0, 255]}

# Instantiate an example
Example((input0, output0))
