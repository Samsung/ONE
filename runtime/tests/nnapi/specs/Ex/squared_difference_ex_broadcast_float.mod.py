# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 1, 2, 1}")

i3 = Output("op3", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model = model.Operation("SQUARED_DIFFERENCE_EX", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [4.0, 8.0, 10.0, 14.0],
          i2: # input 1
          [1.0, 2.0]}

output0 = {i3: # output 0
           [9.0, 36.0, 81.0, 144.0]}

# Instantiate an example
Example((input0, output0))
