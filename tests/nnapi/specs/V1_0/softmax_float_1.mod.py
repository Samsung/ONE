# model
model = Model()

i1 = Input("input", "TENSOR_FLOAT32", "{1, 4}") # batch = 1, depth = 1
beta = Float32Scalar("beta", 0.000001)
output = Output("output", "TENSOR_FLOAT32", "{1, 4}")

# model 1
model = model.Operation("SOFTMAX", i1, beta).To(output)

# Example 1. Input in operand 0,
input0 = {i1: [1., 2., 10., 20.]}

output0 = {output: [.25, .25, .25, .25]}

# Instantiate an example
Example((input0, output0))
