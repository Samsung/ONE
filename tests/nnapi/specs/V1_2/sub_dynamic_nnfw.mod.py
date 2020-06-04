# model
#
# a bit twisted test
# One Sub is enough but this test uses two Subs to check multiple ops
#
#     reshape                   input (i2)
#        |                       | |
#        | dynamic tensor        | |
#        |                       | |
#       sub ---------------------+ |
#        | t1 : dynamic tensor     +
#        |                         |
#       sub -----------------------+
#        | o1 : dynamic tensor
#
#

import dynamic_tensor

model = Model()

model_input1_shape = [3, 4]   # first input shape of Sub. 12 float32s

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input1_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput() # first input of Sub is dynamic tensor

i2 = Input("op2", "TENSOR_FLOAT32", "{1, 4}") # second input of Sub. 4 float32s
t1 = Internal("op3", "TENSOR_FLOAT32", "{}") # result of first Sub. dynamic and shape is not known
act = Int32Scalar("act", 0) # an int32_t scalar activation
o1 = Output("op3", "TENSOR_FLOAT32", "{3, 4}")

model = model.Operation("SUB", test_node_input, i2, act).To(t1) # first Sub
model = model.Operation("SUB", t1, i2, act).To(o1)              # second Sub

model_input1_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
model_input2_data = [0.1, 0.1, 0.1, 0.1]

input0 = {
      dynamic_layer.getModelInput(): model_input1_data,   # input 1
      dynamic_layer.getShapeInput() : model_input1_shape,

      i2: model_input2_data # input 2
      }

output0 = {
      o1: [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8, 11.8]
           }

# Instantiate an example
Example((input0, output0))
