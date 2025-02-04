# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{4}") # a vector of 4 float32s
i2 = Output("op2", "TENSOR_FLOAT32", "{4}") # a vector of 4 float32s
model = model.Operation("EXP", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [3.0, 4.0, 5.0, 6.0]}

output0 = {i2: # output 0
           [20.085537, 54.59815, 148.41316, 403.4288]}

# Instantiate an example
Example((input0, output0))
