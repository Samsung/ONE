# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT16", "{1, 2, 2, 1}")
i2 = Parameter("op2", "TENSOR_INT32", "{4, 2}", [0, 0, 1, 1, 1, 1, 0, 0])
i3 = Output("op3", "TENSOR_FLOAT16", "{1, 4, 4, 1}")
model = model.Operation("PAD", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0,
           3.0, 4.0,]}

output0 = {i3: # output 0
           [0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0]}

# Instantiate an example
Example((input0, output0))
