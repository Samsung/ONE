# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{4}") # a vector of input
i2 = Output("op2", "TENSOR_FLOAT32", "{4}") # a vector of output
model = model.Operation("SQRT", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [62.0, 5.0, 73.0, 34.0]}

output0 = {i2: # output 0
           [7.87400787, 2.23606798, 8.54400375, 5.83095189]}

# Instantiate an example
Example((input0, output0))
