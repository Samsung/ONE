# ------ broadcast test when dim is not 1 -------
tensor_shape_gen = []
tensor_value_gen = []

# input tensors
# tensor name: left
# tflite::interpreter.tensor(1) -> tensor_value_gen[0]
tensor_shape_gen.append('{3, 2, 4}')
tensor_value_gen.append([2.2774236202, -2.4773113728, -0.4044751823, -0.8101355433, -1.9691983461, 2.2676842213, -2.2757787704, -0.8289190531, 0.0121828541, -1.7484937906, -0.5269883871, -0.6346995831, 2.4886128902, -1.5107979774, -0.7372134924, -0.5374289751, -1.2039715052, 1.5278364420, 0.8248311877, -2.4172706604, 0.6997106671, -0.8929677606, 0.3650484681, 1.3652951717, ])

# input tensors
# tensor name: right
# tflite::interpreter.tensor(2) -> tensor_value_gen[1]
tensor_shape_gen.append('{4}')
tensor_value_gen.append([2.2774236202, 0.0121828541, -1.2039715052, -1.9691983461, ])

# output tensors
# tensor name: output
# tflite::interpreter.tensor(0) -> tensor_value_gen[2]
tensor_shape_gen.append('{3, 2, 4}')
tensor_value_gen.append([5.1866583824, -0.0301807225, 0.4869765937, 1.5953176022, -4.4846987724, 0.0276268665, 2.7399728298, 1.6323059797, 0.0277455188, -0.0213016439, 0.6344789863, 1.2498493195, 5.6676259041, -0.0184058305, 0.8875840306, 1.0583041906, -2.7419531345, 0.0186134093, -0.9930732250, 4.7600855827, 1.5935375690, -0.0108788963, -0.4395079613, -2.6885368824, ])

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
