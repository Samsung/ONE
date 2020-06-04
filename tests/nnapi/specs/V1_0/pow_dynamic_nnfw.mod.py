#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright (C) 2018 The Android Open Source Project
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

import dynamic_tensor

# a bit twisted test
# One pow is enough but this test uses two pows to check multiple ops
#
#     reshape                   input (i2)
#        |                       | |
#        | dynamic tensor        | |
#        |                       | |
#       pow ---------------------+ |
#        | t1 : dynamic tensor     +
#        |                         |
#       pow -----------------------+
#        | o1: dynamic tensor

model = Model()

model_input_base_shape = [2, 2]   # first input shape of pow. 4 float32s

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_base_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput() # first input of pow is dynamic tensor

# model definition
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2}") # second input of pow. 4 float32s
t1 = Internal("op3", "TENSOR_FLOAT32", "{}"); # result of first pow. dynamic and shape is not known
o1 = Output("op4", "TENSOR_FLOAT32", "{2, 2}")

model = model.Operation("POW", test_node_input, i2).To(t1) # first pow
model = model.Operation("POW", t1, i2).To(o1)              # second pow

model_input_base_data = [2., 9., 4., 5.]
exponent_data = [1., 2., 2., 3.]
output_data = [2., 6561., 256. , 1953125]

input0 = {
      dynamic_layer.getModelInput(): model_input_base_data,   # input 1
      dynamic_layer.getShapeInput() : model_input_base_shape,

      i2: exponent_data # input 2
      }

output0 = {
      o1: output_data
      }

# Instantiate an example
Example((input0, output0))
