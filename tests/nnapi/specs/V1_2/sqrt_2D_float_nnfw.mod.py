# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2}")
i2 = Output("op2", "TENSOR_FLOAT32", "{2, 2}")
model = model.Operation("SQRT", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
           [36, 2, 9, 12]}
output0 = {i2: # output 0
           [6.0, 1.41421356, 3.0, 3.46410162]}
# Instantiate an example
Example((input0, output0))
