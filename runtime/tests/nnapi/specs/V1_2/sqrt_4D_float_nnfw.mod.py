# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
i2 = Output("op2", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
model = model.Operation("SQRT", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
           [36, 90, 43, 36, 2, 22, 19, 10, 9, 80, 40, 90, 15, 56, 18, 12]}
output0 = {i2: # output 0
           [6.0, 9.48683298, 6.55743852, 6.0, 1.41421356, 4.69041576, 4.35889894, 3.16227766,
            3.0, 8.94427191, 6.32455532, 9.48683298, 3.87298335, 7.48331477, 4.24264069, 3.46410162]}
# Instantiate an example
Example((input0, output0))
