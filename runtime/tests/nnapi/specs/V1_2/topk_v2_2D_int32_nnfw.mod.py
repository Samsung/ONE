# model
model = Model()
i1 = Input("op1", "TENSOR_INT32", "{3,4}") # a vector of input
k = Int32Scalar("k", 2)
i2 = Output("op2", "TENSOR_INT32", "{3,2}") # indexes of output
i3 = Output("op3", "TENSOR_INT32", "{3,2}") # values of output
model = model.Operation("TOPK_V2", i1, k).To([i2, i3])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [40000, 41000, 50000, 60000,
          70000, 80000, 90000, 79000,
          170000, 180000, 190000, 110000]}

output0 = {i2: # output 0
           [60000, 50000,
           90000, 80000,
           190000, 180000],
           i3: # output 1
           [3, 2,
           2, 1,
           2, 1]}

# Instantiate an example
Example((input0, output0))
