# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
w = Int32Scalar("width", 3)
h = Int32Scalar("height", 3)
model = model.Operation("RESIZE_BILINEAR", i1, w, h).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 1.0, 2.0, 2.0]}
output0 = {i2: # output 0
           [1.0, 1.0, 1.0,
            1.666666667, 1.666666667, 1.666666667,
            2.0, 2.0, 2.0]}

# Instantiate an example
Example((input0, output0))
