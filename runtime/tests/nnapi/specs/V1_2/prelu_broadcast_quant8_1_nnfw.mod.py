# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 3}, 1.0f, 2") # a vector of input
i2 = Input("op2", "TENSOR_QUANT8_ASYMM", "{1, 1, 3}, 1.0f, 1") # a vector of alpha
i3 = Output("op3", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 3}, 0.5f, 3") # a vector of output
model = model.Operation("PRELU", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 1, 1,
           2, 2, 2,
           3, 3, 3,
           1, 2, 3],
          i2: # input 1
          [0, 1, 2]}

output0 = {i3: # output 0
           [5, 3, 1,
            3, 3, 3,
            5, 5, 5,
            5, 3, 5]}
# Instantiate an example
Example((input0, output0))
