import dynamic_tensor

# a bit twisted test
# One Add is enough but this test uses two Adds to check multiple ops
#
#     reshape                   input (i2)
#        |                       | |
#        | dynamic tensor        | |
#        |                       | |
#       add ---------------------+ |
#        | t1 : dynamic tensor     +
#        |                         |
#       add -----------------------+
#        | o1: dynamic tensor
#
# Log:
#
# [Linear] * OP_SEQ {cpu}   OpSequence IN(0,1) -> { 0(Reshape:0,1:2) 1(Add:2,3:5) 2(Add:5,3:6) } -> OUT(6)
# [StaticInferer] Operand #5, Dynamic, shape : {1}
# [StaticInferer] Operand #1, Static, shape : {2}
# [StaticInferer] Operand #6, Dynamic, shape : {3 4}
# [StaticInferer] Operand #0, Static, shape : {3 4}
# [StaticInferer] Operand #2, Dynamic, shape : {1}
# [StaticInferer] Operand #3, Static, shape : {1 4}
# [StaticInferer] Operand #4, Static, shape : {}

model = Model()

model_input1_shape = [3, 4]   # first input shape of Add. 12 float32s

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input1_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput() # first input of Add is dynamic tensor

# model definition
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 4}") # second input of Add. 4 float32s
t1 = Internal("op3", "TENSOR_FLOAT32", "{}") # result of first Add. dynamic and shape is not known
act = Int32Scalar("act", 0) # an int32_t scalar activation
o1 = Output("op3", "TENSOR_FLOAT32", "{3, 4}")

model = model.Operation("ADD", test_node_input, i2, act).To(t1) # first add
model = model.Operation("ADD", t1, i2, act).To(o1)              # second add

model_input1_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
model_input2_data = [0.1, 0.1, 0.1, 0.1]

input0 = {
      dynamic_layer.getModelInput(): model_input1_data,   # input 1
      dynamic_layer.getShapeInput() : model_input1_shape,

      i2: model_input2_data # input 2
      }

output0 = {
      o1: [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2]
           }

# Instantiate an example
Example((input0, output0))
