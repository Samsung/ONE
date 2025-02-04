# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 1, 1, 6}")
i2 = Output("op2", "TENSOR_FLOAT32", "{2, 1, 1, 6}")
model = model.Operation("ROUND_EX", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0.001, 8.001, 0.999, 9.999, 0.5, -0.001, -8.001,
                      -0.999, -9.999, -0.5, -2.5, 1.5]}

output0 = {i2: # output 0
           [0.0, 8.0, 1.0, 10.0, 0.0, 0.0, -8.0, -1.0, -10.0, -0.0, -2.0, 2.0]}

# Instantiate an example
Example((input0, output0))
