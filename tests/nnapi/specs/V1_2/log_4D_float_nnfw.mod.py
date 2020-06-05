# model
model = Model()

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 1, 2, 2}")
i3 = Output("op3", "TENSOR_FLOAT32", "{2, 1, 2, 2}")
model = model.Operation("LOG", i1).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [12.0, 36.0, 2.0, 90, 1.0, 0.012, 0.001, 5]}

output0 = {i3: # output 0
           [2.4849066497880004, 3.58351893845611, 0.6931471805599453, 4.499809670330265,
            0.0, -4.422848629194137, -6.907755278982137, 1.6094379124341003]}

# Instantiate an example
Example((input0, output0))
