# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 3, 1}")
i2 = Parameter("op2", "TENSOR_INT32", "{4, 2}", [0, 0, 0, 2, 1, 3, 0, 0])
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 4, 7, 1}")
model = model.Operation("PAD", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0,
           4.0, 5.0, 6.0]}

output0 = {i3: # output 0
           [0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

# Instantiate an example
Example((input0, output0))
