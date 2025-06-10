# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")

i2 = Output("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model = model.Operation("TANH", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-1, 0, 1, 10]}

output0 = {i2: # output 0
           [-.761594156, 0, .761594156, 0.999999996]}

# Instantiate an example
Example((input0, output0))
