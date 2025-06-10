# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 2}")

i3 = Output("op3", "TENSOR_BOOL8", "{2, 2}")
model = model.Operation("NOT_EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0.2, 3.21, 2.4, 7.44],
          i2: # input 1
          [0.21, 7.44]}

output0 = {i3: # output 0
           [True, True, True, False]}

# Instantiate an example
Example((input0, output0))
