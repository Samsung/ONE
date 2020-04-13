# model
model = Model()

i1 = Input("input", "TENSOR_FLOAT32", "{2, 5}") # batch = 2, depth = 5
beta = Float32Scalar("beta", 1.)
output = Output("output", "TENSOR_FLOAT32", "{2, 5}")

# model 1
model = model.Operation("SOFTMAX", i1, beta).To(output)

# Example 1. Input in operand 0,
input0 = {i1:
          [1., 2., 3., 4., 5.,
           -1., -2., -3., -4., -5.]}

output0 = {output:
           [0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647,
            0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231]}

# Instantiate an example
Example((input0, output0))
