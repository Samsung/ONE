# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # a vector of input
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # a vector of alpha
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # a vector of output
model = model.Operation("PRELU", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [3.0, -2.0,
           -1.0, -2.0
	  ],
          i2: # input 1
          [0.0, 1.0,
	   1.0, 2.0]}

output0 = {i3: # output 0
           [3.0, -2.0,
            -1.0, -4.0
           ]}
# Instantiate an example
Example((input0, output0))
