# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{4}") # a vector of input
k = Int32Scalar("k", 2)
i2 = Output("op2", "TENSOR_FLOAT32", "{2}") # values of output
i3 = Output("op3", "TENSOR_INT32", "{2}") # indexes of output
model = model.Operation("TOPK_V2", i1, k).To([i2, i3])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [5.123456789123456789, 3.123456789123456789, 4.123456789123456789, 6.123456789123456789]}

output0 = {i2: # output 0
           [6.123456789123456789, 5.123456789123456789],
           i3: # output 1
           [3, 0]}

# Instantiate an example
Example((input0, output0))
