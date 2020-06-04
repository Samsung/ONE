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
Testing Rsqrt op when the input is dynamic.
      input [2, 2, 2, 2]  shape [4]  (value of shape will be [2, 2, 2, 2])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [2, 2, 2, 2] at execution time
          |
         Rsqrt
          |
        output (dynamic tensor, [2, 2, 2, 2] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [2, 2, 2, 2]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

# write RSQRT test. input is `test_input`

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{2, 2, 2, 2}")

model.Operation("RSQRT", test_node_input).To(model_output)

model_input_data = [1.0, 36.0, 2.0, 90, 4.0, 16.0, 25.0, 100.0,
           23.0, 19.0, 40.0, 256.0, 4.0, 43.0, 8.0, 36.0]
model_output_data = [1.0, 0.166667, 0.70710678118, 0.105409, 0.5, 0.25, 0.2, 0.1,
            0.208514, 0.229416, 0.158114, 0.0625, 0.5, 0.152499, 0.35355339059, 0.166667]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
