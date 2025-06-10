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

# refer to tanh_v1_dynamic.mod.py about the structore

# This adds reshape as the first op in a model and
# returns output of reshape, which is dynamic tensor

'''
Testing Pow op when the input is dynamic.

      input [2, 3]  shape [2]  (value of shape will be [2, 3])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [2, 3] at execution time
          |
         Pow
          |
        output (dynamic tensor, [2, 3] at execution time)
'''

import dynamic_tensor

model = Model()

model_input1_shape = [2, 3]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input1_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

i2 = Input("op2", "TENSOR_FLOAT32", "{2, 3}")
o1 = Output("op3", "TENSOR_FLOAT32", "{2, 3}")

model = model.Operation("POW", test_node_input, i2).To(o1) # Pow

model_input1_data = [1., 2., 3., 4., 5., 6.]
model_input2_data = [1., 2., 3., 0.5, 5., 2.]

input0 = {
      dynamic_layer.getModelInput(): model_input1_data,   # input 1
      dynamic_layer.getShapeInput() : model_input1_shape,

      i2: model_input2_data # input 2
      }

output0 = {
      o1: [1., 4., 27., 2., 3125., 36.]
           }

# Instantiate an example
Example((input0, output0))
