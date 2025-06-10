# model
model = Model()

i1 = Input("input", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
i2 = Output("output", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
model = model.Operation("ABS", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 1.99, -1.4, 0.0001, 0.0002, 16.0, 25.0, 100.0,
	   23.0, 19.0, -40.0, 15.0, 4.0, -43.0, -0.35355339059, 0.35355339059]}

output0 = {i2: # output 0
           [1.0, 1.99, 1.4, 0.0001, 0.0002, 16.0, 25.0, 100.0,
	   23.0, 19.0, 40.0, 15.0, 4.0, 43.0, 0.35355339059, 0.35355339059]}

# Instantiate an example
Example((input0, output0))
