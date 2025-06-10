# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{1, 1, 3, 4}")
axis0 = Int32Scalar("axis0", 0)

i3 = Input("op3", "TENSOR_FLOAT32", "{1, 2, 1, 1}")
i4 = Parameter("op4", "TENSOR_INT32", "{4}", [1, 2, 1, 1])

i5 = Internal("op5", "TENSOR_FLOAT32", "{1, 1, 3, 4}")
i6 = Internal("op6", "TENSOR_FLOAT32", "{1, 2, 1, 1}")

i7 = Output("op7", "TENSOR_FLOAT32", "{1, 2, 3, 4}")

act = Int32Scalar("act", 0)

model = model.Operation("CONCATENATION", i1, axis0).To(i5) # Actually NOP
model = model.Operation("RESHAPE", i3, i4).To(i6) # Actually NOP
model = model.Operation("ADD", i5, i6, act).To(i7)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          i3: # input 1
          [100, 200]}

output0 = {i7: # output 0
           [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212]}

# Instantiate an example
Example((input0, output0))
