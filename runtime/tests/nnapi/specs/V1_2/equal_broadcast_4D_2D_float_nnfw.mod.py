# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3, 2, 2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2}")

i3 = Output("op3", "TENSOR_BOOL8", "{3, 2, 2, 2}")
model = model.Operation("EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [4.89, 11.0, 9.75, 10.20,
           8.25, 2.0, 1.15, 0.0,
           3.0, 1.0, 8.25, 6.0,
           8.45, 3.0, 8.25, 1.2,
           0.0, 3.0, 2.0, 7.34,
           4.3, 9.56, 11.0, 3.0],
          i2: # input 1
          [8.25, 3.0, 2.0, 10.20]}

output0 = {i3: # output 0
           [False, False, False, True,
            True, False, False, False,
            False, False, False, False,
            False, True, False, False,
            False, True, True, False,
            False, False, False, False]
          }

# Instantiate an example
Example((input0, output0))
