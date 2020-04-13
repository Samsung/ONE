# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3}") # a vector of input
i2 = Input("op2", "TENSOR_FLOAT32", "{3}") # a vector of input
i3 = Output("op3", "TENSOR_BOOL8", "{3}") # a vector of output
model = model.Operation("EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [2.0, 3.254232, 5.1232],
          i2: # input 1
          [2.0, 3.254111, 5.1232]}

output0 = {i3: # output 0
           [True, False, True]}

# Instantiate an example
Example((input0, output0))
