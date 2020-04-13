# model
model = Model()

i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{2, 5}, 0.5f, 0") # batch = 2, depth = 5
beta = Float32Scalar("beta", 1.)
output = Output("output", "TENSOR_QUANT8_ASYMM", "{2, 5}, 0.00390625f, 0")

# model 1
model = model.Operation("SOFTMAX", i1, beta).To(output)

# Example 1. Input in operand 0,
input0 = {i1:
          [1, 2, 3, 4, 5,
           255, 254, 253, 252, 251]}

output0 = {output:
           [15, 24, 40, 67, 110,
            110, 67, 40, 24, 15]}

# Instantiate an example
Example((input0, output0))
