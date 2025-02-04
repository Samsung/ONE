# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3, 2, 2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2}")

i3 = Output("op3", "TENSOR_BOOL8", "{3, 2, 2, 2}")
model = model.Operation("NOT_EQUAL", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [4.25, 11.0, 2.2, 10.3,
           8.5, 2.1, 1.0, 0.5,
           3.1, 1.0, 8.5, 6.5,
           11.2, 3.0, 8.5, 1.0,
           0.3, 3.0, 2.1, 7.5,
           4.3, 9.2, 11.1, 3.0],
          i2: # input 1
          [8.5, 3.0, 2.1, 10.3]}

output0 = {i3: # output 0
           [True, True, True, False,
            False, True, True, True,
            True, True, True, True,
            True, False, True, True,
            True, False, False, True,
            True, True, True, True]
          }

# Instantiate an example
Example((input0, output0))
