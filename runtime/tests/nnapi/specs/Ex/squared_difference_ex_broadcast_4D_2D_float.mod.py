# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3, 2, 2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2}")

i3 = Output("op3", "TENSOR_FLOAT32", "{3, 2, 2, 2}")
model = model.Operation("SQUARED_DIFFERENCE_EX", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [74.0, 63.0, 103.0, 88.0,
	   57.0, 9.0, 68.0, 20.0,
	   121.0, 38.0, 54.0, 119.0,
	   56.0, 106.0, 98.0, 98.0,
	   7.0, 89.0, 108.0, 104.0,
	   20.0, 81.0, 4.0, 124.0],
          i2: # input 1
          [56.0, 115.0, 115.0, 116.0]}

output0 = {i3: # output 0
           [324.0, 2704.0, 144.0, 784.0,
	    1.0, 11236.0, 2209.0, 9216.0,
	    4225.0, 5929.0, 3721.0, 9.0,
	    0.0, 81.0, 289.0, 324.0,
	    2401.0, 676.0, 49.0, 144.0,
	    1296.0, 1156.0, 12321.0, 64.0]}

# Instantiate an example
Example((input0, output0))
