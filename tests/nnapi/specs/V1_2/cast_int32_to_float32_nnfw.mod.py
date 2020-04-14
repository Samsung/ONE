# model
model = Model()
i1 = Input("op1", "TENSOR_INT32", "{2, 3}")
i2 = Output("op2", "TENSOR_FLOAT32", "{2, 3}")
model = model.Operation("CAST", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [100, 200, 300, 400, 500, 600]}

output0 = {i2: # output 0
           [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]}

# Instantiate an example
Example((input0, output0))
