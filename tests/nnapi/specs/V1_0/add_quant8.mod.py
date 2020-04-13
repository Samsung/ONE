# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{2}, 2.0, 0")
i2 = Input("op2", "TENSOR_QUANT8_ASYMM", "{2}, 1.0, 0")
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_QUANT8_ASYMM", "{2}, 1.0, 0")
model = model.Operation("ADD", i1, i2, act).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2],
          i2: # input 1
          [3, 4]}

output0 = {i3: # output 0
           [5, 8]}

# Instantiate an example
Example((input0, output0))
