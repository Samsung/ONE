model = Model()
i1 = Input("input", "TENSOR_FLOAT16", "{1, 2, 2, 1}")
perms = Parameter("perms", "TENSOR_INT32", "{4}", [0, 2, 1, 3])
output = Output("output", "TENSOR_FLOAT16", "{1, 2, 2, 1}")

model = model.Operation("TRANSPOSE", i1, perms).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0,
           3.0, 4.0]}

output0 = {output: # output 0
          [1.0, 3.0,
           2.0, 4.0]}

# Instantiate an example
Example((input0, output0))
