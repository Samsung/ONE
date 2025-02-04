# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2}")
i3 = Output("op3", "TENSOR_FLOAT32", "{2, 2}")
model = model.Operation("RSQRT", i1).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 4.0, 8.0]}

output0 = {i3: # output 0
           [1.0,
            0.70710678118,
            0.5,
            0.35355339059]}

# Instantiate an example
Example((input0, output0))
