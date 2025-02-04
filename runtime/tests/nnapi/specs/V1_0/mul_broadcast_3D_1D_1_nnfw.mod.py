# ------ file: report/tensor_mul_1.log ------
tensor_shape_gen = []
tensor_value_gen = []

# input tensors
# tensor name: left
# tflite::interpreter.tensor(1) -> tensor_value_gen[0]
tensor_shape_gen.append('{3, 1, 4}')
tensor_value_gen.append([0.8364028931, 0.6620308161, 1.1811592579, 0.4827561378, -0.3846270740, -1.7236120701, 3.5318591595, 0.2959995866, 1.6260499954, -0.7885181308, -0.8246002197, -1.1367146969, ])

# input tensors
# tensor name: right
# tflite::interpreter.tensor(2) -> tensor_value_gen[1]
tensor_shape_gen.append('{4}')
tensor_value_gen.append([0.8364028931, -0.3846270740, 1.6260499954, 0.6620308161, ])

# output tensors
# tensor name: output
# tflite::interpreter.tensor(0) -> tensor_value_gen[2]
tensor_shape_gen.append('{3, 1, 4}')
tensor_value_gen.append([0.6995698214, -0.2546349764, 1.9206240177, 0.3195994496, -0.3217031956, 0.6629478931, 5.7429795265, 0.1959608495, 1.3600329161, 0.3032854199, -1.3408411741, -0.7525401711, ])

# --------- tensor shape and value defined above ---------
# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", tensor_shape_gen[0])
i2 = Input("op2", "TENSOR_FLOAT32", tensor_shape_gen[1])
act = Int32Scalar("act", 0) # an int32_t scalar fuse_activation
i3 = Output("op3", "TENSOR_FLOAT32", tensor_shape_gen[2])
model = model.Operation("MUL", i1, i2, act).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          tensor_value_gen[0],
          i2: # input 1
          tensor_value_gen[1]}

output0 = {i3: # output 0
           tensor_value_gen[2]}

# Instantiate an example
Example((input0, output0))
