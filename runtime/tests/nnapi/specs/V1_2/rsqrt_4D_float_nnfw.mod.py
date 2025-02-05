# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
i3 = Output("op3", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
model = model.Operation("RSQRT", i1).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 36.0, 2.0, 90, 4.0, 16.0, 25.0, 100.0,
	   23.0, 19.0, 40.0, 256.0, 4.0, 43.0, 8.0, 36.0]}

output0 = {i3: # output 0
           [1.0, 0.166667, 0.70710678118, 0.105409, 0.5, 0.25, 0.2, 0.1,
            0.208514, 0.229416, 0.158114, 0.0625, 0.5, 0.152499, 0.35355339059, 0.166667]}

# Instantiate an example
Example((input0, output0))
