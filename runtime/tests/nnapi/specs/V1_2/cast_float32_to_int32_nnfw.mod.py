# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 3}")
i2 = Output("op2", "TENSOR_INT32", "{2, 3}")
model = model.Operation("CAST", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [100.0, 20.0, 3.0, 0.4, 0.999, 1.1]}

output0 = {i2: # output 0
           [100, 20, 3, 0, 0, 1]}

# Instantiate an example
Example((input0, output0))
