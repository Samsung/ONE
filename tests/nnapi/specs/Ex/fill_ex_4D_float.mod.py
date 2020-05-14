# model
model = Model()
i1 = Input("op1", "TENSOR_INT32", "{4}")
value = Input("op2", "TENSOR_FLOAT32", "{1}")
i2 = Output("op3", "TENSOR_FLOAT32", "{2, 1, 1, 6}")
model = model.Operation("FILL_EX", i1, value).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
            [2, 1, 1, 6],
          value: [1.2]}

output0 = {i2: # output 0
           [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]}

# Instantiate an example
Example((input0, output0))
