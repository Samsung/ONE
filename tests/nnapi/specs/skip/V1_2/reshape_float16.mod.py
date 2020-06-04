# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT16", "{1, 1, 3, 3}") # a line of 3 pixels, 3 components/pixel
i2 = Parameter("op2", "TENSOR_INT32", "{1}", [-1]) # another vector of 2 float32s
i3 = Output("op3", "TENSOR_FLOAT16", "{9}")
model = model.Operation("RESHAPE", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3,
           4, 5, 6,
           7, 8, 9]}

output0 = {i3: # output 0
           [1, 2, 3, 4, 5, 6, 7, 8, 9]}

# Instantiate an example
Example((input0, output0))
