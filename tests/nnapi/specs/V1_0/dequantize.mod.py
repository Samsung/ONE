# model
model = Model()
i1 = Input("op1",  "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 1.f, 0")
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model = model.Operation("DEQUANTIZE", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 32, 128, 255]}

output0 = {i2: # output 0
           [0.0, 32.0, 128.0, 255.0]}

# Instantiate an example
Example((input0, output0))
