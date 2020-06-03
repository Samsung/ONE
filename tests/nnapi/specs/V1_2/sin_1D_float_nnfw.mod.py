# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{4}") #A vector of inputs
i2 = Output("op2", "TENSOR_FLOAT32", "{4}") #A vector of outputs
model = model.Operation("SIN", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
           [2.0, 90.0, 1.0, 0.012]}
output0 = {i2: # output 0
           [0.909297427, 0.893996664, 0.841470985, 0.011999712]}
# Instantiate an example
Example((input0, output0))
