#
# Copyright (C) 2018 The Android Open Source Project
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# refer to tanh_v1_dynamic.mod.py about the structore
#
# This adds reshape as the first op in a model and
# returns output of reshape, which is dynamic tensor
#
# a bit twisted test
# One Mul is enough but this test uses two Mul to check multiple ops
#
#     reshape                   input (i2)
#        |                       | |
#        | dynamic tensor        | |
#        |                       | |
#       Mul ---------------------+ |
#        | t1 : dynamic tensor     +
#        |                         |
#       Mul -----------------------+
#        | o1: dynamic tensor
#

import dynamic_tensor

model = Model()

model_input1_shape = [3, 4]   # first input shape of Mul. 12 float32s

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input1_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput() # first input of Mul is dynamic tensor

# model
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 4}") # second input of Mul. 4 float32s
t1 = Internal("op3", "TENSOR_FLOAT32", "{1}") # result of first Mul. dynamic and shape is not known
act = Int32Scalar("act", 0) # an int32_t scalar activation
o1 = Output("op3", "TENSOR_FLOAT32", "{3, 4}")

model = model.Operation("MUL", test_node_input, i2, act).To(t1) # first Mul
model = model.Operation("MUL", t1, i2, act).To(o1)              # second Mul

model_input1_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
model_input2_data = [2, 2, 2, 2]

input0 = {
      dynamic_layer.getModelInput(): model_input1_data,   # input 1
      dynamic_layer.getShapeInput() : model_input1_shape,

      i2: model_input2_data # input 2
      }

output0 = {
      o1: [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0]
           }

# Instantiate an example
Example((input0, output0))
