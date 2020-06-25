# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{2, 2}, 1.f, 0")
i2 = Input("op2", "TENSOR_QUANT8_ASYMM", "{1, 2}, 1.f, 0")

i3 = Output("op3", "TENSOR_BOOL8", "{2, 2}, 1.f, 0")
model = model.Operation("NOT_EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [7, 3, 2, 7],
          i2: # input 1
          [0, 7]}

output0 = {i3: # output 0
           [True, True, True, False]}

# Instantiate an example
Example((input0, output0))
