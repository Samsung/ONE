# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{4}") #A vector of inputs
i2 = Output("op2", "TENSOR_FLOAT32", "{4}") #A vector of outputs
model = model.Operation("RSQRT", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
           [36.0, 90.0, 43.0, 36.0]}
output0 = {i2: # output 0
           [0.166667, 0.105409, 0.152499, 0.166667]}
# Instantiate an example
Example((input0, output0))
