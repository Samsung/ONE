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

'''
Testing Fully Connected op when the input is dynamic.

      input [3 , 1]  shape [2]  (value of shape will be [3, 1])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [3, 1] at execution time
          |
    Fully Connected
          |
        output (dynamic tensor, [3, 1] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [3, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

weights = Parameter("op2", "TENSOR_FLOAT32", "{1, 1}", [2])

bias = Parameter("b0", "TENSOR_FLOAT32", "{1}", [4])

output = Output("op3", "TENSOR_FLOAT32", "{3, 1}")

act = Int32Scalar("act", 0)

model = model.Operation("FULLY_CONNECTED", test_node_input, weights, bias, act).To(output)

model_input_data = [2, 32, 16]

# Example 1. Input in operand 0,
input0 = {
        dynamic_layer.getModelInput(): model_input_data,
        dynamic_layer.getShapeInput() : model_input_shape,
        }
output0 = {output: # output 0
               [8, 68, 36]}

# Instantiate an example
Example((input0, output0))
