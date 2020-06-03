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
#
#
'''
Testing Minimum op when the input is dynamic.

      input1 [1, 4]  shape [2]  (value of shape will be [1, 4])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 4] at execution time
          |
          |         input2 [2, 4]
          |             |
          +-------------+
          |
        Minimum
          |
        output (dynamic tensor, [2, 4] at execution time)
'''

import dynamic_tensor

model = Model()

input1_shape = [1, 4]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input1_shape, "TENSOR_FLOAT32")

input1 = dynamic_layer.getTestNodeInput()

input2 = Input("intput", "TENSOR_FLOAT32", "{2, 4}")
output = Output("output", "TENSOR_FLOAT32", "{2, 4}")

model = model.Operation("MINIMUM", input1, input2).To(output)

input1_data = [1, 2, 3, 4]

input2_data = [2, 4, 8, 10, 
               -2, -4, -8, -10]

output_data = [1, 2, 3, 4,
               -2, -4, -8, -10]

input0 = {
      dynamic_layer.getModelInput() : input1_data,   # input 1
      dynamic_layer.getShapeInput() : input1_shape,

      input2 : input2_data # input 2
      }

output0 = {
      output: output_data
           }

# Instantiate an example
Example((input0, output0))
