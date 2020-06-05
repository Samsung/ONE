model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 0.5f, 2")
axis = Parameter("axis", "TENSOR_INT32", "{1}", [1])
output = Output("output", "TENSOR_INT32", "{1, 2, 1}")

model = model.Operation("ARGMAX", i1, axis).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 4,
           2, 3]}

output0 = {output: # output 0
          [1,
           0]}

# Instantiate an example
Example((input0, output0))
