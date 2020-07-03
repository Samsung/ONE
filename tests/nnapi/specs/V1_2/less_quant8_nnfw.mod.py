# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{3}, 1.f, 0") # a vector of input
i2 = Input("op2", "TENSOR_QUANT8_ASYMM", "{3}, 1.f, 0") # a vector of input
i3 = Output("op3", "TENSOR_BOOL8", "{3}") # a vector of output
model = model.Operation("LESS", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [2, 3, 0],
          i2: # input 1
          [2, 9, 0]}

output0 = {i3: # output 0
           [False, True, False]}

# Instantiate an example
Example((input0, output0))
