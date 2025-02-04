# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3}")
i2 = Input("op2", "TENSOR_FLOAT32", "{3}")

i3 = Output("op3", "TENSOR_FLOAT32", "{3}")
model = model.Operation("SQUARED_DIFFERENCE_EX", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [2.0, 10.8, 19.4212],
          i2: # input 1
          [2.0, 4.4, 15.9856]}

output0 = {i3: # output 0
           [0.0, 40.96, 11.80334736]}

# Instantiate an example
Example((input0, output0))
