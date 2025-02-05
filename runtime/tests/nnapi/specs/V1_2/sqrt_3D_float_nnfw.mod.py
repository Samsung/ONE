# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2}")
i2 = Output("op2", "TENSOR_FLOAT32", "{2, 2, 2}")
model = model.Operation("SQRT", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
           [90, 36, 22, 10, 9, 80, 40, 18]}
output0 = {i2: # output 0
           [9.48683298, 6.0, 4.69041576, 3.16227766,
            3.0, 8.94427191, 6.32455532, 4.24264069]}
# Instantiate an example
Example((input0, output0))
