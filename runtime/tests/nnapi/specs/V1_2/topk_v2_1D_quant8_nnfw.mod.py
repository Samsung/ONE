# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{4}, 0.5f, 1") # a vector of input
k = Int32Scalar("k", 2)
i2 = Output("op2", "TENSOR_QUANT8_ASYMM", "{2}, 0.5f, 1") # values of output
i3 = Output("op3", "TENSOR_INT32", "{2}") # indexes of output
model = model.Operation("TOPK_V2", i1, k).To([i2, i3])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [7, 4, 5, 6]}

output0 = {i2: # output 0
           [7, 6],
           i3: # output 1
           [0, 3]}

# Instantiate an example
Example((input0, output0))
