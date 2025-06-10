# model
model = Model()
i1 = Input("op1", "TENSOR_INT32", "{3,4}") # a vector of 12 int32s
i2 = Input("op2", "TENSOR_INT32", "{2}") # another vector of 2 int32s
axis = Int32Scalar("axis", 0)
i3 = Output("op3", "TENSOR_INT32", "{2,4}") # a vector of 8 int32s
model = model.Operation("GATHER", i1, axis, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [40000, 41000, 50000, 60000,
          70000, 80000, 90000, 79000,
          170000, 180000, 190000, 110000],
          i2: # input 1
          [2, 0]}

output0 = {i3: # output 0
           [170000, 180000, 190000, 110000,
           40000, 41000, 50000, 60000]}

# Instantiate an example
Example((input0, output0))
