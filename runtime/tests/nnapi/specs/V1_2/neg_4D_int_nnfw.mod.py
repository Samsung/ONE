# model
model = Model()
i1 = Input("op1", "TENSOR_INT32", "{2,3,2,2}")
i2 = Output("op2", "TENSOR_INT32", "{2,3,2,2}")
model = model.Operation("NEG", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [3, 4, 5, 6,
	   -7, 8,-9, 1,
           2, 18, 19, 11,
	   -40, 42, 4, 12,
	   22, -32, 62, 52,
	   92, 59, 69, -312]}

output0 = {i2: # output 0
          [-3, -4, -5, -6,
           7, -8, 9, -1,
           -2, -18, -19, -11,
           40, -42, -4, -12,
	   -22, 32, -62, -52,
	   -92, -59, -69, 312]}

# Instantiate an example
Example((input0, output0))
