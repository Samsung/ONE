# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 1, 2, 2}")
i3 = Output("op3", "TENSOR_FLOAT32", "{2, 1, 2, 2}")
model = model.Operation("SIN", i1).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [12.0, 36.1, 2.0, 90, 1.0, 0.012, 0.001, 5]}

output0 = {i3: # output 0
           [-0.536572918,	-0.999599143,	0.909297427,	0.893996664,
           0.841470985,	0.011999712,	0.001,	-0.958924275]}


# Instantiate an example
Example((input0, output0))
