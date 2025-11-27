# model
model = Model()

i1 = Input("input", "TENSOR_FLOAT32", "{10}")
i2 = Output("output", "TENSOR_FLOAT32", "{10}")
model = model.Operation("ABS", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0.778,	-0.48, -241,	0.9118, -0.466,
          -30.29, -0.4951,	-0.4460, 0.555,
          0.11310]}

output0 = {i2: # output 0
           [0.778,	0.48, 241,	0.9118, 0.466,
          30.29, 0.4951, 0.4460, 0.555,
          0.11310]}

# Instantiate an example
Example((input0, output0))
