# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{3,4}") # a matirx of input
k = Int32Scalar("k", 2)
o1 = Output("op2", "TENSOR_FLOAT32", "{3,2}") # values of output
o2 = Output("op3", "TENSOR_INT32", "{3,2}") # indexes of output
model = model.Operation("TOPK_V2", i1, k).To([o1, o2])

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [3.123456789123456789, 4.123456789123456789, 5.123456789123456789, 6.123456789123456789,
          7.123456789123456789, 8.123456789123456789, 9.123456789123456789, 1.123456789123456789,
          2.123456789123456789, 18.123456789123456789, 19.123456789123456789, 11.123456789123456789]}

output0 = {o1: # output 1
           [6.123456789123456789, 5.123456789123456789,
           9.123456789123456789, 8.123456789123456789,
           19.123456789123456789, 18.123456789123456789],
           o2: # output 1
           [3, 2,
           2, 1,
           2, 1]}

# Instantiate an example
Example((input0, output0))
