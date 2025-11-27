# model
model = Model()

i1 = Input("input", "TENSOR_FLOAT32", "{2, 3, 2}")
i2 = Output("output", "TENSOR_FLOAT32", "{2, 3, 2}")
model = model.Operation("ABS", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0.735078,	-0.46738, -241,	0.9118, -0.46686,
          -3150.219, -0.495291,	-0.42874460, 0.5005046655,
          0.1131106620, -40.0, 15.0]}

output0 = {i2: # output 0
           [0.735078,	0.46738, 241,	0.9118, 0.46686,
          3150.219, 0.495291,	0.42874460, 0.5005046655,
          0.1131106620, 40.0, 15.0]}

# Instantiate an example
Example((input0, output0))
