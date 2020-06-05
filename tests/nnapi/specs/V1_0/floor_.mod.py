# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
model = model.Operation("FLOOR", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-1.5, -1.0, -0.5, 0.0,
            .5,   1.0,  1.5, 10.2]}

output0 = {i2: # output 0
           [-2.0, -1.0, -1.0, 0.0,
             0.0,  1.0,  1.0, 10]}

# Instantiate an example
Example((input0, output0))
