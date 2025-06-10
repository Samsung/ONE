model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 1, 1, 6}")
radius = Int32Scalar("radius", 20)
bias = Float32Scalar("bias", 0.)
alpha = Float32Scalar("alpha", 1.)
beta = Float32Scalar("beta", .5)
output = Output("output", "TENSOR_FLOAT32", "{1, 1, 1, 6}")

model = model.Operation("LOCAL_RESPONSE_NORMALIZATION", i1, radius, bias, alpha, beta).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-1.1, .6, .7, 1.2, -.7, .1]}

output0 = {output: # output 0
           [-.55, .3, .35, .6, -.35, .05]}

# Instantiate an example
Example((input0, output0))
