# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3,4}") # a vector of 12 float32s
i2 = Input("op2", "TENSOR_INT32", "{1,2}") # another vector of 2 int32s
axis = Int32Scalar("axis", 1)
i3 = Output("op3", "TENSOR_FLOAT32", "{3,1,2}") # a vector of 6 float32s
model = model.Operation("GATHER", i1, axis, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [3.123456789123456789, 4.123456789123456789, 5.123456789123456789, 6.123456789123456789,
          7.123456789123456789, 8.123456789123456789, 9.123456789123456789, 1.123456789123456789,
          2.123456789123456789, 18.123456789123456789, 19.123456789123456789, 11.123456789123456789],
          i2: # input 1
          [1, 0]}

output0 = {i3: # output 0
           [4.123456789123456789, 3.123456789123456789,
            8.123456789123456789, 7.123456789123456789,
            18.123456789123456789, 2.123456789123456789]}

# Instantiate an example
Example((input0, output0))
