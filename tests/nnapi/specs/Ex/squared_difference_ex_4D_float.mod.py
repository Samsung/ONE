# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2, 2, 2}")

i3 = Output("op3", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
model = model.Operation("SQUARED_DIFFERENCE_EX", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [4.0, 7.8, 3.1432, 28.987456,
           9.0, 6.0, 5.0, 12.0,
           2.0, 10.8, 19.4212, 10.0,
	   27.0, 36.0, 64.0, 128.0],
          i2: # input 1
          [2.0, 3.2, 1.9856, 8.167952,
	   1.0, 2.0, 5.0, 14.0,
	   2.0, 4.4, 15.9856, 6.0,
	   18.0, 30.0, 61.0, 120.0]}

output0 = {i3: # output 0
           [4.0, 21.16, 1.34003776, 433.451746806016,
            64.0, 16.0, 0, 4.0,
            0.0, 40.96, 11.80334736, 16.0,
	    81.0, 36.0, 9.0, 64.0]}

# Instantiate an example
Example((input0, output0))
