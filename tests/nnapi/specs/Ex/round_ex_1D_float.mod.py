# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{6}")
i2 = Output("op2", "TENSOR_FLOAT32", "{6}")
model = model.Operation("ROUND_EX", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [8.5, 0.0, 3.5, 4.2, -3.5, -4.5]}

output0 = {i2: # output 0
           [8.0, 0.0, 4.0, 4.0, -4.0, -4.0]}

# Instantiate an example
Example((input0, output0))
