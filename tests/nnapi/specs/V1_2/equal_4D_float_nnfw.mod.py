# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # a vector of input
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # a vector of input
i3 = Output("op3", "TENSOR_BOOL8", "{1, 2, 2, 1}") # a vector of output
model = model.Operation("EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1543.25454532, 5.1232, 10.1],
          i2: # input 1
          [0, 5313.25414521, 5.1, 10.1]}

output0 = {i3: # output 0
           [True, False, False, True]}

# Instantiate an example
Example((input0, output0))
