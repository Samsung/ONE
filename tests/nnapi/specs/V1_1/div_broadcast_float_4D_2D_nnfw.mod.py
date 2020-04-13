# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3, 2, 2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2}")
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_FLOAT32", "{3, 2, 2, 2}")
model = model.Operation("DIV", i1, i2, act).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [16, 11, 23, 3, 9, 14, 9, 2, 0, 23, 13, 2, 13, 17, 16, 10, 15, 19, 12, 16, 15, 20, 9, 7],
          i2: # input 1
          [16, 7, 23, 16]}

output0 = {i3: # output 0
           [1.0, 1.57142857, 1.0, 0.1875, 0.5625, 2.0, 0.39130435, 0.125,
            0, 3.28571429, 0.56521739, 0.125, 0.8125, 2.42857143, 0.69565217, 0.625,
            0.9375, 2.71428571, 0.52173913, 1.0, 0.9375, 2.85714286, 0.39130435, 0.4375]
          }

# Instantiate an example
Example((input0, output0))
