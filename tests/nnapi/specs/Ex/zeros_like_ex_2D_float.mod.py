# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 3}")
i2 = Output("op2", "TENSOR_FLOAT32", "{2, 3}")
model = model.Operation("ZEROS_LIKE_EX", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]}

output0 = {i2: # output 0
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

# Instantiate an example
Example((input0, output0))
