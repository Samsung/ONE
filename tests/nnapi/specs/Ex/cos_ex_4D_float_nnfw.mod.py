# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 1, 2, 2}")
i3 = Output("op3", "TENSOR_FLOAT32", "{2, 1, 2, 2}")
model = model.Operation("COS_EX", i1).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [12.0, 36.1, 2.0, 90, 1.0, 0.012, 0.001, 5]}

output0 = {i3: # output 0
           [0.843853959, -0.028311733, -0.416146837, -0.448073616, 
               0.540302306, 0.999928001, 0.999999500, 0.283662185]}

# Instantiate an example
Example((input0, output0))
